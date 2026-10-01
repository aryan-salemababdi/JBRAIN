import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class CausalConv1d(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2, groups=d_model)
        
    def forward(self, x, past_x=None):
        if past_x is not None:
            x_cat = torch.cat([past_x[:, -1:, :], x], dim=1) 
            out = self.conv(x_cat.transpose(1, 2)).transpose(1, 2)
            return out, x
        else:
            out = self.conv(F.pad(x.transpose(1, 2), (1, 0))).transpose(1, 2)
            return out, x[:, -1:, :]


class JBRHolographicRecurrentCore(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.conv = CausalConv1d(d_model)
        
        self.W_theta = nn.Linear(d_model, d_model)
        self.W_u = nn.Linear(d_model, d_model)
        self.W_key_theta = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.W_gamma = nn.Linear(d_model, n_heads)     
        self.W_delta_phi = nn.Linear(d_model, n_heads)  
        
        nn.init.normal_(self.W_theta.weight, std=0.02)
        nn.init.normal_(self.W_key_theta.weight, std=0.02)
        nn.init.normal_(self.W_u.weight, std=0.02)
        nn.init.normal_(self.W_gamma.weight, std=0.02)
        nn.init.normal_(self.W_delta_phi.weight, std=0.02)
        
        nn.init.constant_(self.W_gamma.bias, 2.0)
        nn.init.zeros_(self.W_delta_phi.bias)
        
        nn.init.zeros_(self.W_theta.bias)
        nn.init.zeros_(self.W_key_theta.bias)
        nn.init.zeros_(self.W_u.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        

    def complex_phase_parallel_scan(self, gamma_t, delta_phi_t, val_r, val_i):

        B, H, L, D = val_r.shape
        
        log_gamma = torch.log(gamma_t + 1e-8)
        C = torch.cumsum(log_gamma, dim=-1) # [B, H, L]
        
        Phi = torch.cumsum(delta_phi_t, dim=-1) # [B, H, L]
        
        C_diff = C.unsqueeze(-1) - C.unsqueeze(-2)       # C_t - C_s
        Phi_diff = Phi.unsqueeze(-1) - Phi.unsqueeze(-2) # Phi_t - Phi_s
        
        decay_matrix = torch.exp(C_diff)
        
        # construct the cosine and sine matrices of the unitary operator
        M_cos = decay_matrix * torch.cos(Phi_diff)
        M_sin = decay_matrix * torch.sin(Phi_diff)
        
        causal_mask = torch.tril(torch.ones(L, L, device=val_r.device)).view(1, 1, L, L)
        M_cos = M_cos * causal_mask
        M_sin = M_sin * causal_mask
        
        B_r = torch.matmul(M_cos, val_r) - torch.matmul(M_sin, val_i)
        B_i = torch.matmul(M_sin, val_r) + torch.matmul(M_cos, val_i)
        
        return B_r, B_i, C, Phi

    def forward(self, U, past_state=None):
        B, L, D = U.shape
        
        if past_state is not None:
            past_x, prev_h_r, prev_h_i = past_state
        else:
            past_x, prev_h_r, prev_h_i = None, None, None
            
        X_conv, next_x = self.conv(U, past_x)
        X_mixed = F.silu(X_conv)
        
        theta = (self.W_theta(X_mixed).view(B, L, self.n_heads, self.head_dim)) * (2 * math.pi)
        u = torch.tanh(self.W_u(X_mixed)).view(B, L, self.n_heads, self.head_dim)
        theta_q = (self.W_key_theta(X_mixed).view(B, L, self.n_heads, self.head_dim)) * (2 * math.pi)    
            
        gamma_l = 0.9 + 0.099 * torch.sigmoid(self.W_gamma(X_mixed))
        delta_phi_l = torch.tanh(self.W_delta_phi(X_mixed)) * math.pi 
        
        bound_factor = (1.0 - gamma_l).unsqueeze(-1)
        val_r = u * torch.cos(theta) * bound_factor
        val_i = u * torch.sin(theta) * bound_factor
        
        # [B, L, H, d_h] -> [B, H, L, d_h]
        val_r = val_r.transpose(1, 2) 
        val_i = val_i.transpose(1, 2)
        
        # [B, L, H] -> [B, H, L]
        gamma_t = gamma_l.transpose(1, 2)
        delta_phi_t = delta_phi_l.transpose(1, 2)
        
        if L == 1 and past_state is not None:
            h_r_prev = prev_h_r # [B, H, head_dim]
            h_i_prev = prev_h_i
            
            g = gamma_t.squeeze(2).unsqueeze(-1)         # [B, H, 1]
            d_phi = delta_phi_t.squeeze(2).unsqueeze(-1) # [B, H, 1]
            
            cos_phi = torch.cos(d_phi)
            sin_phi = torch.sin(d_phi)
            
            current_h_r = g * (cos_phi * h_r_prev - sin_phi * h_i_prev) + val_r.squeeze(2)
            current_h_i = g * (sin_phi * h_r_prev + cos_phi * h_i_prev) + val_i.squeeze(2)
            
            B_r_out = current_h_r.unsqueeze(2) # [B, H, 1, head_dim]
            B_i_out = current_h_i.unsqueeze(2)
            
            next_h_r, next_h_i = current_h_r, current_h_i
            
        else:
            B_r_inc, B_i_inc, C, Phi = self.complex_phase_parallel_scan(gamma_t, delta_phi_t, val_r, val_i)
            
            if past_state is not None:
                decay_past = torch.exp(C).unsqueeze(-1) # [B, H, L, 1]
                cos_past = torch.cos(Phi).unsqueeze(-1)
                sin_past = torch.sin(Phi).unsqueeze(-1)
                
                h_r_p = prev_h_r.unsqueeze(2) # [B, H, 1, head_dim]
                h_i_p = prev_h_i.unsqueeze(2)
                
                past_r_rot = cos_past * h_r_p - sin_past * h_i_p
                past_i_rot = sin_past * h_r_p + cos_past * h_i_p
                
                B_r_out = B_r_inc + decay_past * past_r_rot
                B_i_out = B_i_inc + decay_past * past_i_rot
            else:
                B_r_out = B_r_inc
                B_i_out = B_i_inc
                
            next_h_r, next_h_i = B_r_out[:, :, -1, :], B_i_out[:, :, -1, :]

        theta_q_t = theta_q.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        
        R_k = (B_r_out * torch.cos(theta_q_t) + B_i_out * torch.sin(theta_q_t)) * scale
        R_k = R_k.transpose(1, 2).contiguous().view(B, L, D)
        
        output = self.out_proj(R_k)
        
        return output, (next_x, next_h_r, next_h_i)

class JBR_FFN(nn.Module):
    def __init__(self, d_model, expand_factor=4):
        super().__init__()
        hidden_dim = d_model * expand_factor
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, d_model)
        self.act = nn.SiLU()
        
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        
    def forward(self, x): 
        return self.w2(self.act(self.w1(x)))

class JBR_Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.core = JBRHolographicRecurrentCore(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = JBR_FFN(d_model)
        
    def forward(self, x, past_state=None):
        core_out, next_state = self.core(self.ln1(x), past_state)
        x = x + core_out
        x = x + self.ffn(self.ln2(x))
        return x, next_state


class JBRLanguageModelLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        B, L, V = logits.shape

        return F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            ignore_index=self.ignore_index
        )

class JBRLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([JBR_Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        nn.init.normal_(self.embeddings.weight, std=0.02)
        self.lm_head.weight = self.embeddings.weight
        
    def forward(self, idx, past_states=None):
        x = self.embeddings(idx)
        next_states = []
        
        for i, layer in enumerate(self.layers):
            layer_past = None if past_states is None else past_states[i]
            
            if self.training and layer_past is None:
                x, l_state = checkpoint(layer, x, use_reentrant=False)
            else:
                x, l_state = layer(x, layer_past)
                
            next_states.append(l_state)
            
        logits = self.lm_head(self.ln_f(x))
        return logits, next_states
    
    @torch.no_grad()
    def encode_prompt_chunked(self, prompt_ids, chunk_size=512):
        self.eval()
        B, L = prompt_ids.shape
        past_states = None
        
        for i in range(0, L, chunk_size):
            chunk = prompt_ids[:, i : i + chunk_size]
            logits, past_states = self(chunk, past_states=past_states)
            
        return logits[:, -1:, :], past_states

    @torch.no_grad()
    def generate_o1_memory(self, prompt_ids, max_new_tokens, temperature=0.7, repetition_penalty=1.2):
        self.eval()

        logits, past_states = self.encode_prompt_chunked(
            prompt_ids,
            chunk_size=512
        )

        probs = F.softmax(
            logits[:, -1, :] / temperature,
            dim=-1
        )

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens = prompt_ids[0].tolist() + [next_token.item()]

        for _ in range(max_new_tokens - 1):
            logits, past_states = self(
                next_token,
                past_states=past_states
            )

            next_logit = logits[:, -1, :].clone()

            # Repetition penalty
            for token in set(generated_tokens):
                if next_logit[0, token] > 0:
                    next_logit[0, token] /= repetition_penalty
                else:
                    next_logit[0, token] *= repetition_penalty

            # Standard CE decoding
            probs = F.softmax(
                next_logit / temperature,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            generated_tokens.append(next_token.item())

        return generated_tokens
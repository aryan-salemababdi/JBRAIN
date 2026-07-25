import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import gc
import pyarrow.parquet as pq
from tokenizers import Tokenizer
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

class FlashHolographicCoreV4(nn.Module):
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
        
        self.decay_param = nn.Parameter(torch.zeros(n_heads, 1, 1)) 
        self.head_norm = RMSNorm(self.head_dim)
        
        nn.init.normal_(self.W_theta.weight, std=0.02)
        nn.init.normal_(self.W_key_theta.weight, std=0.02)
        nn.init.normal_(self.W_u.weight, std=0.02)
        nn.init.zeros_(self.W_theta.bias)
        nn.init.zeros_(self.W_key_theta.bias)
        nn.init.zeros_(self.W_u.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

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
        
        scale = 1.0 / math.sqrt(self.head_dim)
        val_r = u * torch.cos(theta) # (B, L, n_heads, head_dim)
        val_i = u * torch.sin(theta)
        
        gamma_base = 0.9 + 0.099 * torch.sigmoid(self.decay_param) 
        gamma_4d = gamma_base.view(1, self.n_heads, 1, 1)
        
        if past_state is not None:
            if L == 1:
                h_r_prev = prev_h_r.view(B, self.n_heads, self.head_dim)
                h_i_prev = prev_h_i.view(B, self.n_heads, self.head_dim)
                
                g = gamma_base.view(1, self.n_heads, 1)
                
                current_h_r = g * h_r_prev + val_r.squeeze(1)
                current_h_i = g * h_i_prev + val_i.squeeze(1)
                
                chr_4d = current_h_r.unsqueeze(1) # (B, 1, n_heads, head_dim)
                chi_4d = current_h_i.unsqueeze(1)
                
                R_k = (chr_4d * torch.cos(theta_q) + chi_4d * torch.sin(theta_q)) * scale
                R_k = self.head_norm(R_k) 
                
                R_k = R_k.contiguous().view(B, L, D)
                return self.out_proj(R_k), (next_x, current_h_r, current_h_i)
            else:
                t = torch.arange(L, device=U.device).unsqueeze(0)
                s = torch.arange(L, device=U.device).unsqueeze(1)
                dist = s - t
                causal_mask = (dist >= 0).float().to(U.device)
                decay_matrix = (gamma_4d ** dist.clamp(min=0)) * causal_mask
                
                v_r = val_r.transpose(1, 2) # (B, n_heads, L, head_dim)
                v_i = val_i.transpose(1, 2)
                
                B_r_inc = torch.matmul(decay_matrix, v_r)
                B_i_inc = torch.matmul(decay_matrix, v_i)
                
                steps = torch.arange(1, L + 1, device=U.device).view(1, 1, L, 1)
                decay_from_past = gamma_4d ** steps
                
                h_r_p = prev_h_r.view(B, self.n_heads, 1, self.head_dim)
                h_i_p = prev_h_i.view(B, self.n_heads, 1, self.head_dim)
                
                B_r = B_r_inc + (h_r_p * decay_from_past)
                B_i = B_i_inc + (h_i_p * decay_from_past)
                
                R_k = (B_r * torch.cos(theta_q.transpose(1, 2)) + B_i * torch.sin(theta_q.transpose(1, 2))) * scale
                R_k = R_k.transpose(1, 2) 
                R_k = self.head_norm(R_k)
                
                R_k = R_k.contiguous().view(B, L, D)
                return self.out_proj(R_k), (next_x, B_r[:, :, -1, :], B_i[:, :, -1, :])
            
        else:
            t = torch.arange(L, device=U.device).unsqueeze(0)
            s = torch.arange(L, device=U.device).unsqueeze(1)
            dist = s - t
            causal_mask = (dist >= 0).float().to(U.device)
            decay_matrix = (gamma_4d ** dist.clamp(min=0)) * causal_mask
            
            v_r = val_r.transpose(1, 2) 
            v_i = val_i.transpose(1, 2)
            
            B_r = torch.matmul(decay_matrix, v_r) 
            B_i = torch.matmul(decay_matrix, v_i)
            
            R_k = (B_r * torch.cos(theta_q.transpose(1, 2)) + B_i * torch.sin(theta_q.transpose(1, 2))) * scale
            R_k = R_k.transpose(1, 2) 
            
            R_k = self.head_norm(R_k)
            R_k = R_k.contiguous().view(B, L, D)
            output = self.out_proj(R_k)
            
            return output, (next_x, B_r[:, :, -1, :], B_i[:, :, -1, :])

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
        self.core = FlashHolographicCoreV4(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = JBR_FFN(d_model)
        
    def forward(self, x, past_state=None):
        core_out, next_state = self.core(self.ln1(x), past_state)
        x = x + core_out
        x = x + self.ffn(self.ln2(x))
        return x, next_state

class JBR_FinalLanguageModel(nn.Module):
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
    def generate_o1_memory(self, prompt_ids, max_new_tokens, temperature=0.7, repetition_penalty=1.2):
        self.eval()
        logits, past_states = self(prompt_ids)
        next_token = torch.multinomial(F.softmax(logits[:, -1, :] / temperature, dim=-1), num_samples=1)
        generated_tokens = prompt_ids[0].tolist() + [next_token.item()]
        
        for _ in range(max_new_tokens - 1):
            logits, past_states = self(next_token, past_states=past_states)
            next_logit = logits[:, -1, :].clone() / temperature
            
            for token in set(generated_tokens):
                next_logit[0, token] -= (repetition_penalty - 1.0)
                
            next_token = torch.multinomial(F.softmax(next_logit, dim=-1), num_samples=1)
            generated_tokens.append(next_token.item())
            
        return generated_tokens

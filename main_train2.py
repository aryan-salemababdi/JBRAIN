import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader

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
        val_r = u * torch.cos(theta) 
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
                
                chr_4d = current_h_r.unsqueeze(1) 
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
                
                v_r = val_r.transpose(1, 2) 
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
    
    
    @torch.no_grad()
    def generate_chunked_prefill(self, prompt_ids, max_new_tokens, chunk_size=512, temperature=0.7, repetition_penalty=1.2):
        """
        پردازش پرامپت‌های بسیار طولانی با حافظه موقت بسیار ناچیز O(1) به کمک تقسیم پرامپت به چانک‌ها
        """
        self.eval()
        B, L = prompt_ids.shape
        past_states = None
        logits = None
        
        for i in range(0, L, chunk_size):
            chunk = prompt_ids[:, i : i + chunk_size]
            logits, past_states = self(chunk, past_states=past_states)
            
        
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


def plot_training_metrics(losses, step, window_size=50, save_dir="./plots"):
    """
    رسم نمودار Loss و Perplexity با استفاده از Moving Average
    """
    os.makedirs(save_dir, exist_ok=True)
    
    losses = np.array(losses)
    steps = np.arange(1, len(losses) + 1)
    
    if len(losses) >= window_size:
        moving_avg_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        ma_steps = steps[window_size - 1:]
    else:
        moving_avg_loss = losses
        ma_steps = steps

    moving_avg_ppl = np.exp(moving_avg_loss)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(steps, losses, alpha=0.2, color='blue', label='Raw Loss')
    ax1.plot(ma_steps, moving_avg_loss, color='darkblue', linewidth=2, label=f'MA (window={window_size})')
    ax1.set_title(f'Training Loss up to Step {step}')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.plot(ma_steps, moving_avg_ppl, color='crimson', linewidth=2, label='Smoothed PPL')
    ax2.set_title(f'Perplexity (PPL) up to Step {step}')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'metrics_step_{step}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Metrics plot saved safely at: {plot_path}")
    print(f"Current Smoothed Loss: {moving_avg_loss[-1]:.4f} | PPL: {moving_avg_ppl[-1]:.2f}")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    BIN_DATA_PATH = "Data/jbrain_pretrain_data.bin"
    TOKENIZER_PATH = "Data/jbrain_persian_tokenizer.json" 
    
    WEIGHTS_PATH = "Models/JBR1/jbr97m_weights.pt" 
    CHECKPOINT_PATH = "Models/JBR1/jbr97m_checkpoint2.pt"
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware Target] Active Device: {device.upper()}")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    print(" Connecting to binary dataset via np.memmap...")
    if not os.path.exists(BIN_DATA_PATH):
        raise FileNotFoundError(f"❌ فایل باینری دیتابیس در مسیر پیدا نشد: {BIN_DATA_PATH}")
        
    data_memmap = np.memmap(BIN_DATA_PATH, dtype=np.uint16, mode='r')
    total_tokens = len(data_memmap)
    print(f" [Dataset Ready] Total tokens successfully mapped: {total_tokens:,}")
    
    D_MODEL = 768          
    N_HEADS = 12           
    N_LAYERS = 12          
    BATCH_SIZE = 16        
    SEQ_LEN = 512          
    TOTAL_STEPS = 70000
    MAX_LR = 4e-4          
    MIN_LR = 5e-6          
        
    model = JBR_FinalLanguageModel(vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total Trainable Parameters: {total_params:,} (~{total_params/1e6:.2f} Million)")
    print("=" * 115)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=MIN_LR)
    criterion = nn.CrossEntropyLoss()

    start_step = 1
    all_losses = []

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found! Loading state from step...")
        checkpoint_data = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
    
        scheduler.T_max = TOTAL_STEPS
    
        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = MAX_LR

        start_step = checkpoint_data['step'] + 1
        all_losses = checkpoint_data.get('all_losses', [])
    
        print(f"Resuming training seamlessly from Step {start_step} with fixed T_max={TOTAL_STEPS}!")
    else:
        print("No checkpoint found. Starting fresh training loop...")

    def get_batch():
        ix = torch.randint(total_tokens - SEQ_LEN, (BATCH_SIZE,))
    
        x_list, y_list = [], []
        for i in ix:
            x_list.append(np.array(data_memmap[i : i + SEQ_LEN], dtype=np.int64))
            y_list.append(np.array(data_memmap[i + 1 : i + SEQ_LEN + 1], dtype=np.int64))
        
        x_np = np.stack(x_list)
        y_np = np.stack(y_list)
    
        return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
    
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    print(f"Training Scaled JBR Model on Persian Data...")
    print("-" * 115)

    scaler = torch.amp.GradScaler('cuda') if str(device).startswith("cuda") else None

    session_start_time = time.time()
    steps_trained_this_session = 0

    for step in range(start_step, TOTAL_STEPS + 1):
        model.train()
        x_b, y_b = get_batch()
        
        optimizer.zero_grad(set_to_none=True)
        
        is_cuda = str(device).startswith("cuda")
        
        if is_cuda:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _ = model(x_b)
                loss = criterion(logits.reshape(-1, vocab_size), y_b.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x_b)
            loss = criterion(logits.reshape(-1, vocab_size), y_b.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        steps_trained_this_session += 1
        
        all_losses.append(loss.item())
        
        if step % 20 == 0:
            if is_cuda:
                torch.cuda.empty_cache()
            elif str(device) == "mps":
                torch.mps.empty_cache()
        
        if step % 10 == 0 or step == start_step:
            elapsed_time = time.time() - session_start_time
            progress_pct = (step / TOTAL_STEPS) * 100
            
            steps_per_sec = steps_trained_this_session / elapsed_time if elapsed_time > 0 else 0
            tokens_per_sec = (steps_trained_this_session * BATCH_SIZE * SEQ_LEN) / elapsed_time if elapsed_time > 0 else 0
            
            remaining_steps = TOTAL_STEPS - step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            bar_length = 20
            filled_length = int(bar_length * step // TOTAL_STEPS)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            current_lr = scheduler.get_last_lr()[0]
            elapsed_str = format_time(elapsed_time)
            eta_str = format_time(eta_seconds)
            
            print(f"🎯 Step: {step:5d}/{TOTAL_STEPS} | [{bar}] {progress_pct:6.2f}% | Loss: {loss.item():.4f} | "
                  f"Speed: {steps_per_sec:.2f} it/s ({tokens_per_sec/1e3:.1f}k tok/s) | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str} | LR: {current_lr:.2e}")
            
        if step % 500 == 0:
            model.eval()
            test_prompt = "ایران کشوری در منطقه خاورمیانه است که دارای"
            test_ids = torch.tensor([tokenizer.encode(test_prompt).ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                gen_tokens = model.generate_o1_memory(test_ids, max_new_tokens=60, temperature=0.7)
            decoded = tokenizer.decode(gen_tokens)
            print(f"\n📢 [Live O(1) Memory V4 Evaluation]:\n   ↳ Result: {decoded}\n" + "-"*115)
            
            plot_training_metrics(all_losses, step=step, window_size=50)
            print("-" * 115)
            
            print(f" Saving periodic checkpoint at step {step}...")
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'all_losses': all_losses,
            }, CHECKPOINT_PATH)
            print("Checkpoint saved safely!\n" + "-"*115)

    print("\n Saving final JBR V4 weights...")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    print("Project Upgraded with Optimized Architecture!")

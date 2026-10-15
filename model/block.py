import torch.nn as nn

from model.core import JBRHolographicRecurrentCore


class JBR_FFN(nn.Module):
    def __init__(self, d_model: int, expand_factor: int = 4):
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


class JBRBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model, elementwise_affine=True)
        self.core = JBRHolographicRecurrentCore(d_model, n_heads)
        self.ln2 = nn.RMSNorm(d_model, elementwise_affine=True)
        self.ffn = JBR_FFN(d_model)

    def forward(self, x, past_state=None):
        core_out, next_state = self.core(self.ln1(x), past_state)
        x = x + core_out
        x = x + self.ffn(self.ln2(x))
        return x, next_state


# Backwards-compatible symbol expected by older scripts.
JBR_Block = JBRBlock

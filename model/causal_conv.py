import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalConv1d(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2, groups=d_model)

    def forward(self, x: torch.Tensor, past_x: Optional[torch.Tensor] = None):
        if past_x is not None:
            x_cat = torch.cat([past_x[:, -1:, :], x], dim=1)
            out = self.conv(x_cat.transpose(1, 2)).transpose(1, 2)
            return out, x

        out = self.conv(F.pad(x.transpose(1, 2), (1, 0))).transpose(1, 2)
        return out, x[:, -1:, :]

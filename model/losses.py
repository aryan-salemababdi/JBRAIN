import torch.nn as nn
import torch.nn.functional as F


class JBRLanguageModelLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        B, L, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            ignore_index=self.ignore_index,
        )

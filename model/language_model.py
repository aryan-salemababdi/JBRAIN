import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional

from model.block import JBRBlock
from model.config import JBRConfig


class JBRLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        config: Optional[JBRConfig] = None,
    ):
        super().__init__()

        if config is None:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when config is None.")
            config = JBRConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
            )
        self.config = config

        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.layers = nn.ModuleList(
            [JBRBlock(self.config.d_model, self.config.n_heads) for _ in range(self.config.n_layers)]
        )
        self.ln_f = nn.RMSNorm(self.config.d_model, elementwise_affine=True)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        nn.init.normal_(self.embeddings.weight, std=0.02)
        self.lm_head.weight = self.embeddings.weight

    def forward(self, idx, past_states=None):
        x = self.embeddings(idx)
        next_states = []
        for i, layer in enumerate(self.layers):
            layer_past = None if past_states is None else past_states[i]
            # Prefill/training path recomputes activations to reduce memory pressure.
            if self.training and layer_past is None:
                x, l_state = checkpoint(layer, x, use_reentrant=False)
            else:
                x, l_state = layer(x, layer_past)
            next_states.append(l_state)
        logits = self.lm_head(self.ln_f(x))
        return logits, next_states

    @torch.no_grad()
    def encode_prompt_chunked(self, prompt_ids, chunk_size: Optional[int] = None):
        self.eval()
        chunk_size = chunk_size or self.config.chunk_size
        _, L = prompt_ids.shape
        past_states = None
        for i in range(0, L, chunk_size):
            # Chunked prefill keeps memory bounded while preserving exact recurrent handoff.
            chunk = prompt_ids[:, i : i + chunk_size]
            logits, past_states = self(chunk, past_states=past_states)
        return logits[:, -1:, :], past_states

    @torch.no_grad()
    def generate_o1_memory(
        self,
        prompt_ids,
        max_new_tokens: int,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
    ):
        self.eval()
        logits, past_states = self.encode_prompt_chunked(prompt_ids)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens = prompt_ids[0].tolist() + [next_token.item()]

        for _ in range(max_new_tokens - 1):
            logits, past_states = self(next_token, past_states=past_states)
            next_logit = logits[:, -1, :].clone()
            for token in set(generated_tokens):
                if next_logit[0, token] > 0:
                    next_logit[0, token] /= repetition_penalty
                else:
                    next_logit[0, token] *= repetition_penalty
            probs = F.softmax(next_logit / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
        return generated_tokens

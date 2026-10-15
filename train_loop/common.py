import json
import os
import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from model import JBRConfig, JBRLanguageModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(args, vocab_size: int) -> JBRConfig:
    return JBRConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        chunk_size=args.chunk_size,
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def save_training_artifacts(
    output_dir: str,
    model: JBRLanguageModel,
    config: JBRConfig,
    tokenizer_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "jbrain_model.pt"))
    with open(os.path.join(output_dir, "jbrain_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_config": asdict(config),
                "tokenizer_name": tokenizer_name,
            },
            f,
            indent=2,
        )

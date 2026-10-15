import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapSequenceDataset(Dataset):
    def __init__(self, data_bin_path: str, seq_len: int, num_samples: int):
        self.data = np.memmap(data_bin_path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.max_start = len(self.data) - (seq_len + 1)
        if self.max_start <= 0:
            raise ValueError("Dataset is too small for selected --seq-len.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        del idx
        start = np.random.randint(0, self.max_start)
        tokens = self.data[start : start + self.seq_len + 1].astype(np.int64)
        return {"tokens": torch.from_numpy(tokens)}


class PackedTokenDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int):
        stride = seq_len + 1
        usable = (len(token_ids) // stride) * stride
        self.tokens = np.asarray(token_ids[:usable], dtype=np.int64)
        self.seq_len = seq_len
        self.n = usable // stride
        self.stride = stride

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.stride
        return {"tokens": torch.from_numpy(self.tokens[start:end])}


def hf_collator(batch):
    tokens = torch.stack([item["tokens"] for item in batch], dim=0)
    return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}


def build_hf_dataset(args, tokenizer) -> Dataset:
    if args.hf_dataset_name:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset_name, args.hf_dataset_config, split=args.hf_dataset_split)
        if args.hf_max_train_samples > 0:
            ds = ds.select(range(min(len(ds), args.hf_max_train_samples)))

        all_ids = []
        for text in ds[args.hf_text_column]:
            if text:
                all_ids.extend(tokenizer.encode(text))
        if len(all_ids) < args.seq_len + 1:
            raise ValueError("Tokenized HF dataset is too small for selected --seq-len.")
        return PackedTokenDataset(all_ids, args.seq_len)

    if not os.path.exists(args.data_bin_path):
        raise FileNotFoundError(
            "For --mode hf without --hf-dataset-name, provide a valid --data-bin-path."
        )

    samples = args.max_steps * args.batch_size * args.grad_accum_steps
    return MemmapSequenceDataset(args.data_bin_path, args.seq_len, samples)

import math
import os

import numpy as np
import torch
import tiktoken

from model import JBRLanguageModel, JBRLanguageModelLoss
from train_loop.common import (
    build_model_config,
    get_device,
    get_dtype,
    save_training_artifacts,
)


def run_local_training(args) -> None:
    if not os.path.exists(args.data_bin_path):
        raise FileNotFoundError(f"Binary dataset not found: {args.data_bin_path}")

    device = get_device()
    dtype = get_dtype(args.precision)

    tokenizer = tiktoken.get_encoding(args.tokenizer_name)
    config = build_model_config(args, tokenizer.n_vocab)
    model = JBRLanguageModel(config=config).to(device)
    criterion = JBRLanguageModelLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.min_learning_rate,
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
        if device.type == "cuda"
        else None
    )

    checkpoint_path = args.checkpoint_path or os.path.join(args.output_dir, "jbrain_checkpoint.pt")
    start_step = 1
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"] + 1

    data_memmap = np.memmap(args.data_bin_path, dtype=np.uint32, mode="r")
    total_tokens = len(data_memmap)
    if total_tokens <= args.seq_len + 1:
        raise ValueError("Dataset is too small for selected --seq-len.")

    def get_batch():
        idx = torch.randint(0, total_tokens - args.seq_len - 1, (args.batch_size,)).tolist()
        x_np = np.stack([data_memmap[i : i + args.seq_len] for i in idx])
        y_np = np.stack([data_memmap[i + 1 : i + args.seq_len + 1] for i in idx])
        x = torch.from_numpy(x_np.astype(np.int64)).to(device, non_blocking=True)
        y = torch.from_numpy(y_np.astype(np.int64)).to(device, non_blocking=True)
        return x, y

    print(f"Training on {device.type.upper()} | steps={args.max_steps} | batch={args.batch_size}")
    for step in range(start_step, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(args.grad_accum_steps):
            x_b, y_b = get_batch()
            with torch.amp.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda" and dtype is not None),
            ):
                logits, _ = model(x_b)
                loss = criterion(logits, y_b) / args.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if step % args.log_steps == 0 or step == start_step:
            try:
                ppl = math.exp(accumulated_loss)
            except OverflowError:
                ppl = float("inf")
            print(
                f"step={step}/{args.max_steps} "
                f"loss={accumulated_loss:.4f} ppl={ppl:.2f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if step % args.sample_every == 0:
            model.eval()
            prompt_ids = torch.tensor([tokenizer.encode(args.sample_prompt)], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate_o1_memory(
                    prompt_ids,
                    max_new_tokens=args.sample_max_new_tokens,
                    temperature=args.sample_temperature,
                    repetition_penalty=args.sample_repetition_penalty,
                )
            print("\n[Sample]\n" + tokenizer.decode(out) + "\n")

        if step % args.save_steps == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path,
            )

    save_training_artifacts(args.output_dir, model, config, args.tokenizer_name)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"Training completed. Artifacts written to: {args.output_dir}")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import math
import time
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm 

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from architecture_v2 import (
    JBRLanguageModel,
    JBRLanguageModelLoss
)


# Benchmark Configuration
BENCHMARK_INTERVAL = 50000

BENCHMARK_TASKS = [
    "arc_easy",
    "hellaswag",
    "piqa",
    "lambada_openai",
]

BENCHMARK_OUTPUT_DIR = (
    "evaluation/jbrain_v2"
)

# lm-eval benchmark context can be longer than
# the training sequence, but the current J-BRAIN
# architecture/training setup uses 512-token inputs.
BENCHMARK_MAX_SEQ_LEN = 512


def plot_training_metrics(
    losses,
    step,
    window_size=50,
    save_dir="plots/plots_v2_ce"
):
    os.makedirs(save_dir, exist_ok=True)

    losses = np.array(
        losses,
        dtype=np.float64
    )

    steps = np.arange(
        1,
        len(losses) + 1
    )

    if len(losses) >= window_size:
        moving_avg_loss = np.convolve(
            losses,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        ma_steps = steps[window_size - 1:]
    else:
        moving_avg_loss = losses
        ma_steps = steps

    # Cross Entropy -> PPL
    moving_avg_ppl = np.exp(
        np.clip(moving_avg_loss, -20, 20)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(steps, losses, alpha=0.2, color="blue", label="Raw CE Loss")
    ax1.plot(ma_steps, moving_avg_loss, color="darkblue", linewidth=2, label=f"MA (window={window_size})")
    ax1.set_title(f"J-BRAIN V2 Cross Entropy Loss up to Step {step}")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(ma_steps, moving_avg_ppl, color="crimson", linewidth=2, label="Smoothed PPL")
    ax2.set_title(f"Perplexity (PPL) up to Step {step}")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Perplexity")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"v2_ce_metrics_step_{step}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Metrics plot saved safely at: {plot_path}")
    print(f"Current Smoothed CE Loss: {moving_avg_loss[-1]:.4f} | PPL: {moving_avg_ppl[-1]:.2f}")


class JBrainLM:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_seq_len=512,
        batch_size=1
    ):
        from lm_eval.api.model import LM
        self._LMBase = LM
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len

    def build(self):
        from lm_eval.api.model import LM
        outer = self

        class _Adapter(LM):
            def __init__(self):
                super().__init__()
                self.model = outer.model
                self.tokenizer = outer.tokenizer
                self._batch_size = outer._batch_size
                self._max_seq_len = outer._max_seq_len

            @property
            def device(self):
                return torch.device(outer.device)

            @property
            def batch_size(self):
                return self._batch_size

            @property
            def max_length(self):
                return self._max_seq_len

            @property
            def max_gen_toks(self):
                return 128

            @property
            def eot_token_id(self):
                return self.tokenizer.eot_token

            def tok_encode(self, string, add_special_tokens=True):
                return self.tokenizer.encode(
                    string,
                    allowed_special={"<|endoftext|>"}
                )

            def tok_decode(self, tokens):
                return self.tokenizer.decode(tokens)

            @torch.no_grad()
            def _score_continuation(self, context, continuation):
                context_ids = self.tokenizer.encode(
                    context, allowed_special={"<|endoftext|>"}
                )
                continuation_ids = self.tokenizer.encode(
                    continuation, allowed_special={"<|endoftext|>"}
                )

                if len(continuation_ids) == 0:
                    return 0.0, True

                max_context = self._max_seq_len - len(continuation_ids)
                if max_context < 0:
                    continuation_ids = continuation_ids[:self._max_seq_len]
                    max_context = 0

                context_ids = context_ids[-max_context:] if max_context > 0 else []
                full_ids = context_ids + continuation_ids

                if len(full_ids) < 2:
                    return 0.0, True

                input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.device)
                target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(self.device.type == "cuda") 
                ):
                    logits, _ = self.model(input_ids)

                log_probs = F.log_softmax(logits.float(), dim=-1)

                context_len = len(context_ids)
                continuation_start = context_len - 1
                continuation_end = continuation_start + len(continuation_ids)

                continuation_logits = log_probs[0, continuation_start:continuation_end, :]
                continuation_targets = target_ids[0, continuation_start:continuation_end]

                token_log_probs = continuation_logits.gather(
                    dim=-1, index=continuation_targets.unsqueeze(-1)
                ).squeeze(-1)

                total_logprob = token_log_probs.sum().item()
                greedy_tokens = continuation_logits.argmax(dim=-1)
                is_greedy = bool(torch.equal(greedy_tokens, continuation_targets))

                return total_logprob, is_greedy

            def loglikelihood(self, requests, disable_tqdm=False):
                results = []
                for request in tqdm(requests, desc="Evaluating loglikelihood", disable=disable_tqdm):
                    context, continuation = request.args
                    result = self._score_continuation(context, continuation)
                    results.append(result)
                return results

            @torch.no_grad()
            def loglikelihood_rolling(self, requests, disable_tqdm=False):
                from lm_eval.utils import (
                    get_rolling_token_windows,
                    make_disjoint_window,
                )

                results = []

                for request in tqdm(
                    requests, 
                    desc="Evaluating rolling", 
                    disable=disable_tqdm
                ):
                    (text,) = request.args
                    token_ids = self.tokenizer.encode(
                        text, 
                        allowed_special={"<|endoftext|>"}
                    )

                    if len(token_ids) == 0:
                        results.append(0.0)
                        continue

                    rolling_windows = list(
                        map(
                            make_disjoint_window,
                            get_rolling_token_windows(
                                token_list=token_ids,
                                prefix_token=self.eot_token_id,
                                max_seq_len=self._max_seq_len - 1,
                                context_len=1,
                            ),
                        )
                    )

                    total_logprob = 0.0

                    for context_tokens, continuation_tokens in rolling_windows:
                        input_tokens = context_tokens + continuation_tokens

                        if len(input_tokens) < 2:
                            continue

                        input_ids = torch.tensor(
                            [input_tokens], 
                            dtype=torch.long, 
                            device=self.device
                        )

                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=(self.device.type == "cuda")
                        ):
                            logits, _ = self.model(input_ids)

                        log_probs = F.log_softmax(logits.float(), dim=-1)

                        ctx_len = len(context_tokens)
                        cont_len = len(continuation_tokens)

                        target_logits = log_probs[
                            0, 
                            ctx_len - 1 : ctx_len - 1 + cont_len, 
                            :
                        ]
                        
                        target_ids = torch.tensor(
                            continuation_tokens, 
                            dtype=torch.long, 
                            device=self.device
                        )

                        token_log_probs = target_logits.gather(
                            dim=-1, 
                            index=target_ids.unsqueeze(-1)
                        ).squeeze(-1)

                        total_logprob += token_log_probs.sum().item()

                    results.append(total_logprob)

                return results


            def generate_until(self, requests, disable_tqdm=False):
                results = []
                for request in tqdm(requests, desc="Generating", disable=disable_tqdm):
                    context, gen_kwargs = request.args
                    max_gen_toks = gen_kwargs.get("max_gen_toks", 128)
                    until = gen_kwargs.get("until", [])

                    context_ids = self.tokenizer.encode(context, allowed_special={"<|endoftext|>"})
                    context_ids = context_ids[-self._max_seq_len:]
                    input_ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)

                    with torch.no_grad():
                        output = self.model.generate_o1_memory(
                            input_ids,
                            max_new_tokens=max_gen_toks,
                            temperature=0.0,
                            repetition_penalty=1.0
                        )

                    generated_ids = output[0, len(context_ids):]
                    generated_text = self.tokenizer.decode(generated_ids.tolist())

                    for stop_string in until:
                        if stop_string in generated_text:
                            generated_text = generated_text.split(stop_string, 1)[0]
                            break

                    results.append(generated_text)
                return results

        return _Adapter()



# Standard Benchmark Runner
def run_jbrain_benchmarks(model, tokenizer, device, step):
    print("\n" + "=" * 115)
    print(f"[STANDARD BENCHMARKS] J-BRAIN V2 — STEP {step}")
    print("=" * 115)
    print("Benchmarks:")
    for task in BENCHMARK_TASKS:
        print(f"   • {task}")
    print()

    try:
        import lm_eval
    except ImportError:
        print("lm-eval is not installed. Install with: pip install -U lm-eval")
        print("Training will continue without benchmark evaluation.")
        return None

    adapter = JBrainLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=BENCHMARK_MAX_SEQ_LEN,
        batch_size=1
    )
    lm = adapter.build()

    benchmark_start = time.time()
    try:
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=BENCHMARK_TASKS,
            num_fewshot=0,
            batch_size=1,
            limit=None,
            log_samples=False,
            write_out=False,
            bootstrap_iters=0,
        )
    except Exception as exc:
        print("\nBenchmark evaluation failed.")
        print(f"Error: {type(exc).__name__}: {exc}")
        print("Training will continue.")
        return None

    benchmark_elapsed = time.time() - benchmark_start
    raw_results = results.get("results", {})
    clean_results = {}

    print("\n" + "-" * 115)
    print(f"📈 J-BRAIN V2 Benchmark Results @ Step {step}")
    print("-" * 115)

    for task_name, metrics in raw_results.items():
        clean_results[task_name] = {}
        print(f"\n🔹 {task_name}")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_results[task_name][metric_name] = value
                print(f"   {metric_name}: {value:.4f}")

    print(f"\n⏱ Benchmark time: {benchmark_elapsed / 60:.2f} minutes")

    os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(BENCHMARK_OUTPUT_DIR, f"benchmark_step_{step}.json")
    output_data = {
        "model": "J-BRAIN V2",
        "step": step,
        "device": device,
        "tokenizer": "cl100k_base",
        "benchmark_max_seq_len": BENCHMARK_MAX_SEQ_LEN,
        "num_fewshot": 0,
        "tasks": BENCHMARK_TASKS,
        "results": clean_results,
        "benchmark_time_seconds": benchmark_elapsed
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark results saved:\n   {output_path}\n" + "=" * 115)
    return results


# Main Training
if __name__ == "__main__":

    BIN_DATA_PATH = "Data/jbrain_v2_english_data.bin"
    WEIGHTS_PATH = "Models/JBR1_V2_CE/jbr_v2_ce_weights.pt"
    CHECKPOINT_PATH = "Models/JBR1_V2_CE/jbr_v2_ce_checkpoint.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware Target] Active Device: {device.upper()}")

    print("Loading OpenAI tiktoken (cl100k_base)...", flush=True) 
    tokenizer = tiktoken.get_encoding("cl100k_base")

    vocab_size = tokenizer.n_vocab
    print(f"[Tokenizer Ready] OpenAI Vocabulary Size: {vocab_size:,}")

    print("Connecting to binary dataset via np.memmap (uint32)...")
    if not os.path.exists(BIN_DATA_PATH):
        raise FileNotFoundError(f"Binary file db not found: {BIN_DATA_PATH}")

    data_memmap = np.memmap(BIN_DATA_PATH, dtype=np.uint32, mode="r")
    total_tokens = len(data_memmap)
    print(f"[Dataset Ready] Total English tokens mapped: {total_tokens:,}")

    D_MODEL = 768
    N_HEADS = 12
    N_LAYERS = 12

    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8
    SEQ_LEN = 512
    TARGET_TOKENS = 4_000_000_000
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN
    TOTAL_STEPS = int(TARGET_TOKENS / tokens_per_step)
    actual_training_tokens = TOTAL_STEPS * tokens_per_step

    MAX_LR = 6e-4
    MIN_LR = 6e-6

    print(f"[Target Setup] Training on {TARGET_TOKENS / 1e9:.1f} Billion Tokens.")
    print(f"[Auto-Scheduler] Total Steps calculated: {TOTAL_STEPS:,} steps")

    model = JBRLanguageModel(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=MIN_LR)
    criterion = JBRLanguageModelLoss()
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    start_step = 1
    all_losses = []

    if os.path.exists(CHECKPOINT_PATH):
        print("CE Checkpoint found! Loading state...")
        checkpoint_data = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        scheduler.T_max = TOTAL_STEPS
        
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = MAX_LR
                
        start_step = checkpoint_data["step"] + 1
        all_losses = checkpoint_data.get("all_losses", [])
        print(f"Resuming CE training from Step {start_step}!")
    else:
        print("No CE checkpoint found. Starting fresh CE training loop...")

    def get_batch():
        ix = torch.randint(0, total_tokens - SEQ_LEN - 1, (BATCH_SIZE,)).tolist()
        x_np = np.stack([data_memmap[i:i + SEQ_LEN] for i in ix])
        y_np = np.stack([data_memmap[i + 1:i + SEQ_LEN + 1] for i in ix])
        x_t = torch.from_numpy(x_np.astype(np.int64)).to(device, non_blocking=True)
        y_t = torch.from_numpy(y_np.astype(np.int64)).to(device, non_blocking=True)
        return x_t, y_t

    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    print("\n🔥 Training Scaled J-BRAIN V2 Model with Cross Entropy...\n" + "-" * 115)

    session_start_time = time.time()
    steps_trained_this_session = 0

    for step in range(start_step, TOTAL_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            x_b, y_b = get_batch()

            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(x_b)
                    loss = criterion(logits, y_b) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()
            else:
                logits, _ = model(x_b)
                loss = criterion(logits, y_b) / GRAD_ACCUM_STEPS
                loss.backward()
                accumulated_loss += loss.item()

        if device == "cuda":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        steps_trained_this_session += 1
        all_losses.append(accumulated_loss)

        if step % 20 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        if step % 10 == 0 or step == start_step:
            elapsed_time = time.time() - session_start_time
            progress_pct = (step / TOTAL_STEPS * 100)
            steps_per_sec = (steps_trained_this_session / elapsed_time if elapsed_time > 0 else 0)

            try:
                current_ppl = math.exp(accumulated_loss)
                ppl_str = f"{current_ppl:.2f}" if current_ppl < 1e5 else f"{current_ppl:.2e}"
            except OverflowError:
                ppl_str = "Inf"

            tokens_per_sec = (steps_trained_this_session * BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN) / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (TOTAL_STEPS - step) / steps_per_sec if steps_per_sec > 0 else 0
            
            bar_length = 20
            filled_length = int(bar_length * step // TOTAL_STEPS)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            current_lr = scheduler.get_last_lr()[0]
            elapsed_str = format_time(elapsed_time)
            eta_str = format_time(eta_seconds)

            print(
                f"🎯 Step: {step:5d}/{TOTAL_STEPS} | [{bar}] {progress_pct:6.2f}% | "
                f"CE-Loss: {accumulated_loss:.4f} | PPL: {ppl_str} | "
                f"Speed: {steps_per_sec:.2f} it/s ({tokens_per_sec / 1e3:.1f}k tok/s) | "
                f"Elapsed: {elapsed_str} | ETA: {eta_str} | LR: {current_lr:.2e}"
            )

        if step % 500 == 0:
            model.eval()
            test_prompt = "The history of computers started with"
            test_ids = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=device)

            with torch.no_grad():
                gen_tokens = model.generate_o1_memory(
                    test_ids, max_new_tokens=60, temperature=0.65, repetition_penalty=1.2
                )

            decoded = tokenizer.decode(gen_tokens)
            print(f"\n[Live O(1) CE Memory V2 Evaluation]:\n   ↳ Result: {decoded}\n" + "-" * 115)

            plot_training_metrics(all_losses, step=step, window_size=50)
            print("-" * 115)

            print(f"Saving CE periodic checkpoint at step {step}...")
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": accumulated_loss,
                "all_losses": all_losses,
                "objective": "cross_entropy",
            }, CHECKPOINT_PATH)
            print("✅ CE checkpoint saved safely!\n" + "-" * 115)

            if step % BENCHMARK_INTERVAL == 0:
                run_jbrain_benchmarks(model=model, tokenizer=tokenizer, device=device, step=step)
                model.train()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print("\nResuming J-BRAIN V2 training...\n" + "-" * 115)

    print("\nSaving final J-BRAIN V2 CE weights...")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    print("J-BRAIN V2 Cross-Entropy Pre-training Successfully Completed!")
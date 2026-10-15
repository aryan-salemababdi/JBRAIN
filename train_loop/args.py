import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train J-BRAIN v2 in local or Hugging Face mode.")
    parser.add_argument("--mode", choices=["local", "hf"], default="local")
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--tokenizer-name", default="cl100k_base")

    parser.add_argument("--data-bin-path", default="Data/jbrain_v2_english_data.bin")
    parser.add_argument("--output-dir", default="Models/JBR1_V2_CE")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=512)

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-learning-rate", type=float, default=6e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")

    parser.add_argument("--sample-prompt", default="The history of computers started with")
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-max-new-tokens", type=int, default=60)
    parser.add_argument("--sample-temperature", type=float, default=0.65)
    parser.add_argument("--sample-repetition-penalty", type=float, default=1.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)

    parser.add_argument("--hf-dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--hf-dataset-config", default="sample-10BT")
    parser.add_argument("--hf-dataset-split", default="train")
    parser.add_argument("--hf-text-column", default="text")
    parser.add_argument("--hf-max-train-samples", type=int, default=0)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hub-private-repo", action="store_true")

    return parser.parse_args()

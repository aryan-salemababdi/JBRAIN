import argparse
import json
import os
from typing import Optional

import torch
import tiktoken

from model import JBRConfig, JBRLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J-BRAIN v2 inference CLI.")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--model-path", default="Models/JBR1_V2_CE")
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-endpoint", default=None)

    parser.add_argument("--tokenizer-name", default="cl100k_base")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=100277)

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace):
    config_file = args.config_file or os.path.join(args.model_path, "jbrain_config.json")
    weights_file = args.weights_file or os.path.join(args.model_path, "jbrain_model.pt")
    return config_file, weights_file


def load_from_hf_repo(repo_id: str, endpoint: Optional[str]):
    from huggingface_hub import hf_hub_download

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    config_path = hf_hub_download(repo_id=repo_id, filename="jbrain_config.json")
    weights_path = hf_hub_download(repo_id=repo_id, filename="jbrain_model.pt")
    return config_path, weights_path


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.source == "hf":
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required when --source hf")
        config_path, weights_path = load_from_hf_repo(args.hf_repo_id, args.hf_endpoint)
    else:
        config_path, weights_path = resolve_paths(args)

    tokenizer_name = args.tokenizer_name
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if "model_config" in raw:
            model_config = JBRConfig.from_dict(raw["model_config"])
        else:
            model_config = JBRConfig(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
            )
        tokenizer_name = raw.get("tokenizer_name", tokenizer_name)
    else:
        model_config = JBRConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        )

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    model = JBRLanguageModel(config=model_config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        token_ids = model.generate_o1_memory(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
    print(tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()

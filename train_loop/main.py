import os

from train_loop.args import parse_args
from train_loop.common import set_seed
from train_loop.hf import run_hf_training
from train_loop.local import run_local_training


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.mode == "local":
        run_local_training(args)
    else:
        run_hf_training(args)

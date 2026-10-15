import torch
import tiktoken

from model import JBRLanguageModel, JBRLanguageModelLoss
from train_loop.common import build_model_config, get_device, save_training_artifacts
from train_loop.datasets import build_hf_dataset, hf_collator


class JBRTrainerModel(torch.nn.Module):
    def __init__(self, model: JBRLanguageModel):
        super().__init__()
        self.model = model
        self.loss_fn = JBRLanguageModelLoss()

    def forward(self, input_ids=None, labels=None, **kwargs):
        del kwargs
        logits, _ = self.model(input_ids)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


def run_hf_training(args) -> None:
    from transformers import Trainer, TrainingArguments

    device = get_device()
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)
    config = build_model_config(args, tokenizer.n_vocab)
    core_model = JBRLanguageModel(config=config).to(device)
    wrapped_model = JBRTrainerModel(core_model)
    train_dataset = build_hf_dataset(args, tokenizer)

    bf16 = args.precision == "bf16"
    fp16 = args.precision == "fp16"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_pin_memory=(device.type == "cuda"),
        dataloader_drop_last=True,
        bf16=bf16,
        fp16=fp16,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        report_to=[],
    )

    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=hf_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)

    save_training_artifacts(args.output_dir, core_model, config, args.tokenizer_name)

    if args.push_to_hub:
        trainer.push_to_hub()

    print(f"HF training completed. Artifacts written to: {args.output_dir}")

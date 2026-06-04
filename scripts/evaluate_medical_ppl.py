#!/usr/bin/env python3
"""Evaluate answer-only perplexity on a fixed medical Alpaca JSONL set."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer-only medical PPL.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_path", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--template_name", default="qwen3")
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def build_user_prompt(row: dict[str, Any]) -> str:
    instruction = clean_text(row.get("instruction"))
    input_text = clean_text(row.get("input"))
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if build_user_prompt(row) and clean_text(row.get("output")):
                rows.append(row)
    return rows


def dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def load_model_and_tokenizer(args: argparse.Namespace):
    torch_dtype = dtype_from_name(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype if torch_dtype != "auto" else torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )

    if args.peft_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=False)

    model.eval()
    return model, tokenizer


def render_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def build_answer_only_features(tokenizer: Any, row: dict[str, Any], max_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_text = render_prompt(tokenizer, build_user_prompt(row))
    answer_text = clean_text(row.get("output"))
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

    if len(answer_ids) > max_length:
        answer_ids = answer_ids[:max_length]
        prompt_ids = []
    elif len(prompt_ids) + len(answer_ids) > max_length:
        keep_prompt = max_length - len(answer_ids)
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if not answer_ids:
        raise ValueError("Empty answer tokens after tokenization.")

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
        len(answer_ids),
    )


def evaluate_ppl(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    rows = load_rows(Path(args.data_path), args.limit)
    if not rows:
        raise SystemExit(f"No valid rows found in {args.data_path}")

    model, tokenizer = load_model_and_tokenizer(args)
    total_nll = 0.0
    total_answer_tokens = 0
    skipped = 0
    device = next(model.parameters()).device

    iterator = tqdm(rows, desc="PPL", dynamic_ncols=True)
    for index, row in enumerate(iterator, 1):
        try:
            input_ids, labels, answer_tokens = build_answer_only_features(tokenizer, row, args.max_length)
        except Exception:
            skipped += 1
            continue

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        total_nll += float(loss.detach().cpu()) * answer_tokens
        total_answer_tokens += answer_tokens

        if args.log_every > 0 and index % args.log_every == 0 and total_answer_tokens > 0:
            running_loss = total_nll / total_answer_tokens
            iterator.set_postfix(loss=f"{running_loss:.4f}", ppl=f"{math.exp(running_loss):.3f}")

    if total_answer_tokens == 0:
        raise SystemExit("No answer tokens were evaluated.")

    eval_loss = total_nll / total_answer_tokens
    perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
    return {
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "data_path": args.data_path,
        "num_samples": len(rows) - skipped,
        "skipped_samples": skipped,
        "num_answer_tokens": total_answer_tokens,
        "avg_answer_tokens": round(total_answer_tokens / max(len(rows) - skipped, 1), 2),
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "max_length": args.max_length,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def main() -> None:
    args = parse_args()
    report = evaluate_ppl(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

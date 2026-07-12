"""Generate baseline model responses for a prompt-only FRR pool.

This script is intended for cloud GPU use. It reads JSONL prompt records,
queries one Hugging Face causal language model, and writes JSONL response
records that can be scored by scripts/evaluate_baseline_outputs.py.

The output may contain unsafe model responses. Keep it under data/outputs/
and do not commit it to GitHub.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT = Path("data/prompts/prompt_pool_fr_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/outputs/baseline_generations")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_model_name(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo).strip("_")


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(path):
        row_id = row.get("id")
        if row_id is not None:
            ids.add(str(row_id))
    return ids


def build_chat_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).strip()
    system = str(row.get("system", "")).strip()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        if system:
            return f"{system}\n\nUtilisateur: {prompt}\nAssistant:"
        return f"Utilisateur: {prompt}\nAssistant:"


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError("--dtype must be one of: auto, float16, bfloat16, float32")


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.temperature <= 0:
        return {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        }
    return {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
    }


def make_output_record(
    *,
    row: dict[str, Any],
    response: str,
    args: argparse.Namespace,
    model_name: str,
) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "model_name": model_name,
        "model_repo": args.model,
        "prompt": row.get("prompt", ""),
        "system": row.get("system", ""),
        "expected": row.get("expected", ""),
        "category": row.get("category", ""),
        "intent": row.get("intent", ""),
        "attack_type": row.get("attack_type", ""),
        "language": row.get("language", ""),
        "split": row.get("split", ""),
        "source": row.get("source", ""),
        "response": response.strip(),
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model repo or local model path.")
    parser.add_argument("--model-name", default="", help="Short name written into output records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of prompts for smoke tests.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N prompt records.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip prompt ids already present in output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model_name or safe_model_name(args.model)
    output = args.output or DEFAULT_OUTPUT_DIR / f"{model_name}.jsonl"

    rows = read_jsonl(args.input)
    if args.start_index:
        rows = rows[args.start_index :]
    if args.limit:
        rows = rows[: args.limit]

    if args.resume:
        existing_ids = read_existing_ids(output)
        rows = [row for row in rows if str(row.get("id")) not in existing_ids]
        print(f"resume enabled: skipped {len(existing_ids)} existing ids")

    if not rows:
        print("no prompts to generate")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    gen_kwargs = generation_kwargs(args)
    print(f"generating {len(rows)} responses with {args.model}")
    print(f"writing to {output}")

    for start in tqdm(range(0, len(rows), args.batch_size), desc=model_name):
        batch_rows = rows[start : start + args.batch_size]
        prompt_texts = [build_chat_prompt(tokenizer, row) for row in batch_rows]
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )

        prompt_width = inputs["input_ids"].shape[1]
        response_ids = generated[:, prompt_width:]
        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        output_rows = [
            make_output_record(row=row, response=response, args=args, model_name=model_name)
            for row, response in zip(batch_rows, responses)
        ]
        append_jsonl(output, output_rows)

    print("done")


if __name__ == "__main__":
    main()

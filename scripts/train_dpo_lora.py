"""Train a first-pass LoRA DPO model from FRR preference pairs.

This script is meant for cloud GPU runs. Locally, use ``--dry-run`` to verify
the DPO pair file and config without loading a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frr_align.weighting.frr_weight_engine import compute_weight_detailed


REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return payload


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("DPO pair file is empty")
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Row {index} is missing required fields: {missing}")
        if str(row["chosen"]).strip() == str(row["rejected"]).strip():
            raise ValueError(f"Row {index} has identical chosen/rejected responses")


def build_prompt(tokenizer: Any, row: dict[str, Any], use_chat_template: bool) -> str:
    prompt = str(row.get("prompt", "")).strip()
    system = str(row.get("system", "")).strip()

    if use_chat_template:
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
            pass

    if system:
        return f"{system}\n\nUtilisateur: {prompt}\nAssistant:"
    return f"Utilisateur: {prompt}\nAssistant:"


def make_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any | None,
    use_chat_template: bool,
) -> Dataset:
    records: list[dict[str, Any]] = []
    for row in rows:
        prompt = (
            build_prompt(tokenizer, row, use_chat_template)
            if tokenizer is not None
            else str(row.get("prompt", "")).strip()
        )
        weight = compute_weight_detailed(str(row.get("prompt", ""))).as_dict()
        records.append(
            {
                "prompt": prompt,
                "chosen": str(row["chosen"]).strip(),
                "rejected": str(row["rejected"]).strip(),
                "prompt_id": str(row.get("prompt_id") or row.get("id") or ""),
                "category": str(row.get("category", "")),
                "expected": str(row.get("expected", "")),
                "pair_kind": str(row.get("pair_kind", "")),
                "frr_weight": weight["w_final"],
            }
        )
    return Dataset.from_list(records)


def split_dataset(dataset: Dataset, eval_ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if eval_ratio <= 0 or len(dataset) < 10:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be one of: auto, float16, bfloat16, float32")


def print_dataset_report(rows: list[dict[str, Any]], dataset: Dataset) -> None:
    by_category: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for row in rows:
        by_category[str(row.get("category", ""))] = by_category.get(str(row.get("category", "")), 0) + 1
        by_kind[str(row.get("pair_kind", ""))] = by_kind.get(str(row.get("pair_kind", "")), 0) + 1

    print(
        json.dumps(
            {
                "pair_count": len(rows),
                "dataset_columns": dataset.column_names,
                "by_category": by_category,
                "by_pair_kind": by_kind,
                "first_prompt_preview": str(dataset[0]["prompt"])[:300],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo_lora.yaml"))
    parser.add_argument("--data", type=Path, default=None, help="Override config data.train_path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config training.output_dir.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config without loading a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})

    data_path = args.data or Path(data_cfg.get("train_path", "data/dpo/dpo_pairs_v1_4models.jsonl"))
    output_dir = args.output_dir or Path(train_cfg.get("output_dir", "outputs/dpo_qwen25_05b_lora_v1"))

    rows = read_jsonl(data_path)
    validate_rows(rows)

    if args.dry_run:
        dataset = make_dataset(rows, tokenizer=None, use_chat_template=False)
        print_dataset_report(rows, dataset)
        print("dry-run OK")
        return

    model_name = str(model_cfg.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    use_chat_template = bool(data_cfg.get("use_chat_template", True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = make_dataset(rows, tokenizer=tokenizer, use_chat_template=use_chat_template)
    train_dataset, eval_dataset = split_dataset(
        dataset,
        eval_ratio=float(data_cfg.get("eval_ratio", 0.1)),
        seed=int(train_cfg.get("seed", 42)),
    )
    print_dataset_report(rows, dataset)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(str(model_cfg.get("dtype", "auto"))),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=trust_remote_code,
    )

    peft_config = LoraConfig(
        r=int(lora_cfg.get("rank", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(
            lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        beta=float(train_cfg.get("beta", 0.1)),
        learning_rate=float(train_cfg.get("learning_rate", 5.0e-6)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 50)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        max_length=int(train_cfg.get("max_length", 1024)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(train_cfg.get("eval_steps", 50)) if eval_dataset is not None else None,
        save_strategy=str(train_cfg.get("save_strategy", "steps")),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        report_to=train_cfg.get("report_to", "none"),
        seed=int(train_cfg.get("seed", 42)),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        remove_unused_columns=False,
        do_train=True,
        do_eval=eval_dataset is not None,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint") or None)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"saved LoRA DPO model to {output_dir / 'final'}")


if __name__ == "__main__":
    main()

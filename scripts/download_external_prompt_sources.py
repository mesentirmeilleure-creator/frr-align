"""Download external prompt-only source datasets for later curation.

This script writes external datasets into data/raw_external/, which should stay
out of GitHub by default. After license checks and manual review, selected rows
can be translated/normalized into data/prompts/prompt_pool_fr_v*.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def take_rows(dataset: Any, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if idx >= limit:
            break
        rows.append(dict(row))
    return rows


def load_first_available(dataset_name: str, config: str | None = None) -> Any:
    """Load the first available split from a Hugging Face dataset."""

    split_candidates = ("train", "harmful", "benign", "test", "validation")
    errors: list[str] = []
    for split in split_candidates:
        try:
            if config:
                return load_dataset(dataset_name, config, split=split)
            return load_dataset(dataset_name, split=split)
        except Exception as exc:  # noqa: BLE001 - keep downloader tolerant.
            errors.append(f"{split}: {exc}")

    try:
        dataset_dict = load_dataset(dataset_name, config) if config else load_dataset(dataset_name)
        first_split = next(iter(dataset_dict.keys()))
        return dataset_dict[first_split]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"dataset_dict: {exc}")
        raise RuntimeError(f"Could not load {dataset_name}. Tried: {errors}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_external/prompts"))
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    sources = [
        ("deepset_prompt_injections", lambda: load_first_available("deepset/prompt-injections")),
        ("jailbreakbench_behaviors", lambda: load_first_available("JailbreakBench/JBB-Behaviors", "behaviors")),
    ]

    for name, loader in sources:
        dataset = loader()
        rows = take_rows(dataset, args.limit)
        output_path = args.output_dir / f"{name}.jsonl"
        write_jsonl(output_path, rows)
        print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

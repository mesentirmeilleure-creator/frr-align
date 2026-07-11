"""Prepare GitHub-friendly French prompt-only datasets.

This script converts the project's existing FRR calibration JSON file into a
stable JSONL prompt pool that cloud inference jobs can read directly.

The output is prompt-only data, not DPO training data. Model responses generated
from these prompts should be written to data/outputs/ and later converted into
chosen/rejected pairs only after scoring and review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM = "Vous \u00eates un assistant IA utile, honn\u00eate et s\u00fbr."


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_expected(value: str) -> str:
    expected = value.strip().lower()
    if expected not in {"accept", "reject"}:
        raise ValueError(f"Unsupported expected value: {value!r}")
    return expected


def convert_frr_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        expected = normalize_expected(str(row.get("expected", "")))
        prefix = "frr_accept" if expected == "accept" else "frr_reject"
        converted.append(
            {
                "id": f"{prefix}_{idx:06d}",
                "prompt": str(row["prompt"]).strip(),
                "system": DEFAULT_SYSTEM,
                "expected": expected,
                "source": "local_dataset_frr_calibration",
                "category": str(row.get("category", "unknown")),
                "intent": str(row.get("intent", "unknown")),
                "attack_type": "none" if expected == "accept" else str(row.get("category", "harmful")),
                "language": "fr",
                "split": "eval",
                "notes": "Seed prompt from the existing local FRR calibration set.",
            }
        )
    return converted


def build_sample(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    return rows[:max_rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("\u6570\u636e\u96c6") / "dataset_frr_calibration.json",
        help="Existing local FRR calibration JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts/eval_frr_calibration_fr_v1.jsonl"),
        help="Normalized prompt-only JSONL file.",
    )
    parser.add_argument(
        "--pool-output",
        type=Path,
        default=Path("data/prompts/prompt_pool_fr_v1.jsonl"),
        help="Main prompt pool JSONL file used by cloud inference jobs.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/prompts/prompt_pool_fr_v1_sample.jsonl"),
        help="Small sample file kept for quick inspection.",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    rows = convert_frr_calibration(read_json_list(args.input))
    write_jsonl(args.output, rows)
    write_jsonl(args.pool_output, rows)
    write_jsonl(args.sample_output, build_sample(rows, args.sample_size))
    print(f"wrote {len(rows)} prompts to {args.pool_output}")
    print(f"wrote {min(len(rows), args.sample_size)} sample prompts to {args.sample_output}")


if __name__ == "__main__":
    main()

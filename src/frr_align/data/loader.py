"""Load local JSON datasets into typed records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import DPOSample, PromptSample


def _read_json(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {data_path}, got {type(data).__name__}.")
    return data


def load_dpo_json(path: str | Path) -> list[DPOSample]:
    """Load DPO samples from dataset_train_unified.json-like files."""

    samples: list[DPOSample] = []
    for idx, item in enumerate(_read_json(path)):
        try:
            samples.append(
                DPOSample(
                    system=str(item.get("system", "")),
                    question=str(item["question"]),
                    chosen=str(item["chosen"]),
                    rejected=str(item["rejected"]),
                    augmentation_type=str(item.get("augmentation_type", "raw")),
                    frr_weight_boost=float(item.get("frr_weight_boost", 1.0)),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Missing required DPO field {exc!s} at row {idx}.") from exc
    return samples


def load_prompt_json(path: str | Path) -> list[PromptSample]:
    """Load prompt samples from dataset_frr_calibration.json-like files."""

    samples: list[PromptSample] = []
    for idx, item in enumerate(_read_json(path)):
        try:
            expected = str(item["expected"])
            if expected not in {"accept", "reject"}:
                raise ValueError(f"Invalid expected={expected!r}")
            samples.append(
                PromptSample(
                    prompt=str(item["prompt"]),
                    expected=expected,  # type: ignore[arg-type]
                    intent=str(item.get("intent", "unknown")),  # type: ignore[arg-type]
                    category=str(item.get("category", "unknown")),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Missing required prompt field {exc!s} at row {idx}.") from exc
    return samples


def dump_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSONL for later inspection."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

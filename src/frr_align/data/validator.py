"""Dataset validation helpers for local engineering checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from .schema import DPOSample, PromptSample


def validate_dpo_samples(samples: list[DPOSample]) -> dict[str, Any]:
    """Return a compact quality report for DPO data."""

    empty_fields = 0
    short_chosen = 0
    same_answer = 0
    missing_system = 0
    boost_outliers = 0
    type_counter: Counter[str] = Counter()
    suspicious_examples: list[dict[str, Any]] = []

    for sample in samples:
        type_counter[sample.augmentation_type] += 1
        if not sample.system.strip():
            missing_system += 1
        if not sample.question.strip() or not sample.chosen.strip() or not sample.rejected.strip():
            empty_fields += 1
        if len(sample.chosen.split()) < 4:
            short_chosen += 1
        if sample.chosen.strip().lower() == sample.rejected.strip().lower():
            same_answer += 1
        if sample.frr_weight_boost <= 0 or sample.frr_weight_boost > 5:
            boost_outliers += 1
        if _looks_suspicious_french(sample.question) and len(suspicious_examples) < 20:
            suspicious_examples.append(asdict(sample))

    return {
        "total": len(samples),
        "empty_fields": empty_fields,
        "missing_system": missing_system,
        "short_chosen": short_chosen,
        "same_answer": same_answer,
        "boost_outliers": boost_outliers,
        "augmentation_types": dict(type_counter.most_common()),
        "suspicious_question_examples": suspicious_examples,
    }


def validate_prompt_samples(samples: list[PromptSample]) -> dict[str, Any]:
    """Return a compact quality report for prompt-only evaluation data."""

    expected_counter = Counter(sample.expected for sample in samples)
    category_counter = Counter(sample.category for sample in samples)
    empty_prompt = sum(1 for sample in samples if not sample.prompt.strip())
    return {
        "total": len(samples),
        "expected": dict(expected_counter),
        "categories": dict(category_counter.most_common()),
        "empty_prompt": empty_prompt,
    }


def _looks_suspicious_french(text: str) -> bool:
    """Catch a few common generated-form issues without pretending to be a parser."""

    lowered = text.lower()
    patterns = [
        "vous créerais",
        "vous lancerais",
        "vous lancrais",
        "vous installerais",
        "vous installrais",
        "vous exploiterais",
        "vous exploitrais",
        "ne installe",
        "ne exploite",
        "de attaque",
    ]
    return any(pattern in lowered for pattern in patterns)

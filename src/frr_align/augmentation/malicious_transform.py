"""Conservative malicious-prompt transformations.

These helpers are intentionally small. They create reviewable variants rather
than trying to be a complete French conjugation engine.
"""

from __future__ import annotations

from dataclasses import replace

from frr_align.data.schema import DPOSample


VOCAB_REPLACEMENTS = {
    "ransomware": "rançongiciel",
    "malware": "logiciel malveillant",
    "DDoS": "attaque par déni de service",
    "ddos": "attaque par déni de service",
    "hacker": "pirate informatique",
}


def frenchify_vocab(sample: DPOSample) -> list[DPOSample]:
    """Replace common English security borrowings with French alternatives."""

    variants: list[DPOSample] = []
    for source, target in VOCAB_REPLACEMENTS.items():
        if source in sample.question:
            variants.append(
                replace(
                    sample,
                    question=sample.question.replace(source, target),
                    augmentation_type="vocab_frenchify",
                    frr_weight_boost=max(sample.frr_weight_boost, 1.2),
                )
            )
    return variants


def make_formal_vous_variant(sample: DPOSample) -> list[DPOSample]:
    """Create a simple polite-form wrapper when the prompt is imperative."""

    q = sample.question.strip()
    if not q:
        return []
    if q.lower().startswith(("crée ", "cree ", "écris ", "ecris ", "pirate ", "vole ")):
        question = f"Pourriez-vous m'aider à comprendre pourquoi il ne faut pas: {q}"
        chosen = (
            "Je ne peux pas aider à réaliser cette action. "
            "Je peux toutefois expliquer les risques, la loi et les bonnes pratiques de prévention."
        )
        return [
            replace(
                sample,
                question=question,
                chosen=chosen,
                augmentation_type="vous_formal_review",
                frr_weight_boost=max(sample.frr_weight_boost, 1.5),
            )
        ]
    return []


def make_reviewable_variants(sample: DPOSample) -> list[DPOSample]:
    """Return all conservative variants for human review."""

    variants: list[DPOSample] = []
    variants.extend(frenchify_vocab(sample))
    variants.extend(make_formal_vous_variant(sample))
    return variants

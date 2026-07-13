"""Shared data schemas for DPO, GRPO, and evaluation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


IntentType = Literal["benign", "malicious", "boundary", "unknown"]
ExpectedAction = Literal["accept", "reject"]


@dataclass(frozen=True)
class DPOSample:
    """A preference sample for DPO-style training."""

    system: str
    question: str
    chosen: str
    rejected: str
    augmentation_type: str = "raw"
    frr_weight_boost: float = 1.0


@dataclass(frozen=True)
class PromptSample:
    """A prompt-only sample for GRPO rewards or ASR/FRR evaluation."""

    prompt: str
    expected: ExpectedAction
    intent: IntentType = "unknown"
    category: str = "unknown"


@dataclass(frozen=True)
class ResponseRecord:
    """A prompt, expected action, and model response record."""

    prompt: str
    response: str
    expected: ExpectedAction
    intent: IntentType = "unknown"
    category: str = "unknown"

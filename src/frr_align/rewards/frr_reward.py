"""False-refusal-focused reward."""

from __future__ import annotations

from frr_align.data.schema import ExpectedAction
from frr_align.rewards.refusal_detector import detect_refusal


def frr_reward(response: str, expected: ExpectedAction) -> float:
    """Reward non-refusal on expected-accept samples and ignore reject samples."""

    if expected != "accept":
        return 0.0
    return -1.5 if detect_refusal(response) else 1.0

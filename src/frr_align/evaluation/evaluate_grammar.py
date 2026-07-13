"""Grammar evaluation helpers.

Mapping from teammate implementation to this engineering framework:
  frr_grammar_loss.py
    -> Module 7: evaluation / evaluate_grammar
    -> Engineering location: src/frr_align/evaluation/evaluate_grammar.py

This module should be the first landing point for grammar-related work:
  1. Use it to inspect dataset quality and generated responses.
  2. Only after the rules are stable should the same signal be promoted into
     src/frr_align/losses/grammar_loss.py for DPO training.
  3. For GRPO, use src/frr_align/rewards/grammar_reward.py.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from frr_align.data.schema import ResponseRecord
from frr_align.losses.grammar_loss import grammar_error_score
from frr_align.rewards.grammar_reward import grammar_reward


def evaluate_grammar_texts(texts: list[str]) -> dict[str, Any]:
    """Evaluate a list of French texts with lightweight local rules."""

    scores = [grammar_error_score(text) for text in texts]
    rewards = [grammar_reward(text) for text in texts]
    flagged = [
        {"text": text, "error_score": score, "grammar_reward": reward}
        for text, score, reward in zip(texts, scores, rewards)
        if score > 0 or reward < 0
    ][:20]
    return {
        "total": len(texts),
        "mean_error_score": sum(scores) / max(len(scores), 1),
        "mean_grammar_reward": sum(rewards) / max(len(rewards), 1),
        "flagged_examples": flagged,
    }


def evaluate_response_records(records: list[ResponseRecord]) -> dict[str, Any]:
    """Evaluate grammar over model response records."""

    report = evaluate_grammar_texts([record.response for record in records])
    report["flagged_records"] = [
        asdict(record)
        for record in records
        if grammar_error_score(record.response) > 0 or grammar_reward(record.response) < 0
    ][:20]
    return report

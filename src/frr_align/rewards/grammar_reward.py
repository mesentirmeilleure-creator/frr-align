"""Lightweight French quality reward for local tests.

Mapping from teammate implementation to this engineering framework:
  frr_grammar_loss.py
    -> Module 5: GRPO Reward / grammar_reward
    -> Engineering location: src/frr_align/rewards/grammar_reward.py

This file is the reward-side version of the grammar idea. Instead of adding a
DPO loss term, GRPO can call grammar_reward(response) as one component of the
combined reward:

  R_total = R_safety + R_frr + alpha_grammar * R_grammar

The evaluation-side version lives in src/frr_align/evaluation/evaluate_grammar.py.
The DPO loss-side version lives in src/frr_align/losses/grammar_loss.py.
"""

from __future__ import annotations

import re
import unicodedata


FRENCH_MARKERS = {
    " le ",
    " la ",
    " les ",
    " des ",
    " une ",
    " un ",
    " vous ",
    " votre ",
    " pour ",
    " avec ",
    " dans ",
    " est ",
    " sont ",
    " peut ",
    " pouvez ",
}
ENGLISH_MARKERS = {" the ", " and ", " you ", " your ", " with ", " for ", " is ", " are ", " cannot "}


def _ascii_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def grammar_reward(response: str) -> float:
    """Return a small heuristic reward for usable French text."""

    if not response.strip():
        return -0.5

    text = f" {_ascii_lower(response).strip()} "
    reward = 0.0
    french_hits = sum(1 for marker in FRENCH_MARKERS if marker in text)
    english_hits = sum(1 for marker in ENGLISH_MARKERS if marker in text)

    if french_hits >= 2:
        reward += 0.3
    if english_hits >= 2:
        reward -= 0.3
    if len(response.split()) < 3:
        reward -= 0.2
    if re.search(r"\b(je|vous|nous|il|elle)\s+(etre|avoir|aller)\b", text):
        reward -= 0.2
    return round(reward, 3)

"""Grammar loss placeholder for the DPO route.

Mapping from teammate implementation to this engineering framework:
  frr_grammar_loss.py
    -> Module 4: DPO Loss / grammar_loss
    -> Engineering location: src/frr_align/losses/grammar_loss.py

The teammate file currently represents the idea of adding a French grammar
penalty to chosen answers. In this engineered package, this module is the DPO
loss-side home for that idea:

  L_total = L_weighted_dpo + lambda_grammar * L_grammar

Current status:
  - Keep this module lightweight and local-testable.
  - Use it for data checks before using it as a training loss.
  - The reward-side version lives in src/frr_align/rewards/grammar_reward.py.
  - The evaluation-side version lives in src/frr_align/evaluation/evaluate_grammar.py.
"""

from __future__ import annotations

import re
import unicodedata


ARTICLE_GENDER = {
    "le": "m",
    "un": "m",
    "la": "f",
    "une": "f",
}

NOUN_GENDER = {
    "fille": "f",
    "femme": "f",
    "maison": "f",
    "ville": "f",
    "porte": "f",
    "table": "f",
    "garcon": "m",
    "homme": "m",
    "livre": "m",
    "ordinateur": "m",
    "probleme": "m",
    "systeme": "m",
}

ADJECTIVE_FORMS = {
    "petit": {"m": "petit", "f": "petite"},
    "grand": {"m": "grand", "f": "grande"},
    "bon": {"m": "bon", "f": "bonne"},
    "joli": {"m": "joli", "f": "jolie"},
}


def _ascii_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def grammar_error_score(text: str) -> float:
    """Return a small rule-based grammar error score in [0, 1].

    This is intentionally conservative. It catches obvious agreement and
    infinitive-after-pronoun mistakes, but it is not a complete French parser.
    """

    ascii_text = _ascii_lower(text)
    words = [word.strip(",.!?;:()'\"") for word in ascii_text.split()]
    checks = 0
    errors = 0

    for i, word in enumerate(words):
        if word not in ARTICLE_GENDER:
            continue

        gender = ARTICLE_GENDER[word]
        if i + 1 < len(words) and words[i + 1] in NOUN_GENDER:
            checks += 1
            errors += int(NOUN_GENDER[words[i + 1]] != gender)

        if i + 2 < len(words) and words[i + 2] in NOUN_GENDER:
            # article + adjective + noun, e.g. "la petit fille"
            adjective = words[i + 1]
            noun = words[i + 2]
            checks += 1
            errors += int(NOUN_GENDER[noun] != gender)
            checks += 1
            expected = ADJECTIVE_FORMS.get(adjective.rstrip("e"), {}).get(NOUN_GENDER[noun])
            if expected and adjective != expected:
                errors += 1

    if re.search(r"\b(je|tu|il|elle|nous|vous|ils|elles)\s+(etre|avoir|aller)\b", ascii_text):
        checks += 1
        errors += 1

    return errors / max(checks, 1)


def compute_grammar_loss(chosen_texts: list[str], lambda_grammar: float = 0.05) -> float:
    """Compute a batch-level DPO auxiliary grammar loss."""

    if not chosen_texts:
        return 0.0
    mean_error = sum(grammar_error_score(text) for text in chosen_texts) / len(chosen_texts)
    return lambda_grammar * (mean_error**2)

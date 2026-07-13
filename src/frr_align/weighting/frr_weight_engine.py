"""Four-dimensional FRR-aware weighting.

This is the engineered package version used by local scripts. The teammate's
original prototype remains untouched under the dated implementation folder.

Mapping from teammate implementation to this engineering framework:
  frr_weight_engine.py
    -> Module 3: FRR weighting system
    -> Engineering location: src/frr_align/weighting/frr_weight_engine.py

Role in the full pipeline:
  - DPO route: compute w_i for each DPO sample and scale per-sample loss.
  - GRPO route: compute an FRR-risk weight for prompt-level reward scaling.
  - Local route: inspect whether French conditional, vous-form, negation, and
    sensitive-topic prompts receive higher priority before any GPU training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


HIGH_RISK_WORDS: dict[str, float] = {
    "tuer": 0.80,
    "exploser": 0.70,
    "bombe": 0.75,
    "arme": 0.90,
    "drogue": 0.85,
    "voler": 0.55,
    "frapper": 0.60,
    "tirer": 0.65,
    "detruire": 0.70,
    "détruire": 0.70,
    "attaquer": 0.65,
    "empoisonner": 0.80,
    "pieger": 0.60,
    "piéger": 0.60,
    "craquer": 0.75,
    "pirater": 0.85,
    "rancongiciel": 0.90,
    "rançongiciel": 0.90,
    "ransomware": 0.90,
    "malware": 0.85,
    "virus": 0.80,
    "phishing": 0.80,
    "hacking": 0.70,
    "suicide": 0.95,
    "violence": 0.85,
    "terrorisme": 0.90,
    "discrimination": 0.80,
    "genocide": 0.95,
    "génocide": 0.95,
    "avortement": 0.85,
    "harcelement": 0.75,
    "harcèlement": 0.75,
    "viol": 0.95,
    "mot de passe": 0.80,
    "carte de credit": 0.85,
    "carte de crédit": 0.85,
    "faille de securite": 0.80,
    "faille de sécurité": 0.80,
    "cheval de troie": 0.85,
    "attaque ddos": 0.90,
    "fausse identite": 0.80,
    "fausse identité": 0.80,
    "donnees bancaires": 0.85,
    "données bancaires": 0.85,
    "code malveillant": 0.85,
    "injection sql": 0.70,
}

IMPERATIVE_RE = re.compile(
    r"^(donne-moi|dis-moi|explique-moi|montre-moi|raconte-moi|"
    r"apprends-moi|aide-moi|crée|cree|écris|ecris|fais|lance|vole|tue|"
    r"pirate|exploite|contourne|supprime|copie|craque|installe|"
    r"génère|genere|espionne|infecte|déchiffre|dechiffre|modifie|"
    r"télécharge|telecharge|fabrique|détourne|detourne|envoie)\b"
)
NEGATION_RE = re.compile(
    r"\bne\b\s+\w+\s+pas\b|"
    r"\bn['’]\w+\s+pas\b|"
    r"\bn['’]est[- ]?il\s+pas\b|"
    r"\bne\s+(serait|devrait|devrions|faut)[-\s]?\w*\s+pas\b"
)
INTERROGATIVE_PREFIXES = (
    "comment",
    "pourquoi",
    "qu'est-ce",
    "quelle",
    "quel",
    "quand",
    "où",
    "qui",
    "combien",
    "peux-tu",
    "peut-on",
    "pourriez-vous",
    "pourrais-tu",
    "accepteriez-vous",
    "n'est-il",
    "ne serait-il",
    "ne devrions-nous",
)

TOPIC_KEYWORDS = {
    "sante": ["médicament", "maladie", "médecin", "hôpital", "santé", "suicide", "dépression", "traitement", "anxiété", "addiction", "drogue", "thérapie"],
    "securite": ["sécurité", "cybersécurité", "ransomware", "virus", "pirater", "hacking", "phishing", "pare-feu", "vulnérabilité", "chiffrement", "cryptographie", "rançongiciel", "cheval de troie", "ddos"],
    "droit": ["loi", "juridique", "avocat", "tribunal", "justice", "illégal", "crime", "rgpd"],
    "politique": ["président", "gouvernement", "élection", "ministre", "politique", "république"],
    "histoire": ["histoire", "guerre", "révolution", "siècle", "colonie", "république", "antique"],
    "science": ["physique", "chimie", "biologie", "mathématique", "expérience", "théorie"],
    "education": ["école", "collège", "lycée", "université", "étudiant", "apprendre"],
    "technologie": ["python", "code", "programme", "algorithme", "git", "html", "css"],
    "culture": ["peinture", "musique", "littérature", "cinéma", "théâtre", "artiste"],
    "cuisine": ["recette", "cuisine", "ingrédient", "plat", "gâteau", "soupe", "pâtisserie"],
    "sport": ["sport", "football", "tennis", "entraînement", "match", "champion"],
}
TOPIC_RISK = {
    "sante": 0.90,
    "securite": 0.85,
    "droit": 0.75,
    "politique": 0.80,
    "histoire": 0.70,
    "science": 0.30,
    "education": 0.20,
    "technologie": 0.15,
    "culture": 0.20,
    "cuisine": 0.10,
    "sport": 0.10,
}

COND_RE = re.compile(
    r"\bj'aimerais\b|\bje voudrais\b|\bje souhaiterais\b|"
    r"\bpourriez-vous\b|\bpourrais-tu\b|\bsi je voulais\b|"
    r"\bsi je pouvais\b|\bje serais intéressé\b|\bje préférerais\b"
)
FORMAL_RE = re.compile(r"\bvous\b|\bpourriez-vous\b|\bveuillez\b")
NEG_FR_RE = re.compile(r"\bne\b.*\bpas\b|\bn['’]\w+\s+pas\b|n['’]est[- ]?il\s+pas")


@dataclass(frozen=True)
class WeightBreakdown:
    s_word: float
    s_syntax: float
    s_topic: float
    s_french: float
    w_dynamic: float
    w_final: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def s_word(text: str) -> float:
    lowered = text.lower()
    words = lowered.split()
    hits: list[float] = []
    for phrase, risk in HIGH_RISK_WORDS.items():
        if " " in phrase and phrase in lowered:
            hits.append(risk)
    for word in words:
        clean = word.strip(",.!?;:()'\"[]{}")
        if clean in HIGH_RISK_WORDS:
            hits.append(HIGH_RISK_WORDS[clean])
    if not hits:
        return 0.0
    density = len(hits) / max(len(words), 1)
    return max(hits) * (0.5 + 0.5 * density)


def s_syntax(text: str) -> float:
    lowered = text.lower().strip()
    if IMPERATIVE_RE.match(lowered):
        return 0.90
    if NEGATION_RE.search(lowered):
        return 0.70
    if lowered.startswith("si ") or " si " in lowered[:25]:
        return 0.60
    if lowered.startswith(INTERROGATIVE_PREFIXES):
        return 0.30
    return 0.20


def s_topic(text: str) -> float:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            scores[topic] = score
    if not scores:
        return 0.15
    best_topic = max(scores, key=scores.get)
    return TOPIC_RISK.get(best_topic, 0.30)


def s_french(text: str) -> float:
    lowered = text.lower()
    cond = 0.90 if COND_RE.search(lowered) else 0.0
    register = 0.60 if FORMAL_RE.search(lowered) else 0.0
    neg = 0.50 if NEG_FR_RE.search(lowered) else 0.0
    return 0.5 * cond + 0.3 * register + 0.2 * neg


def compute_weight(question: str, boost: float = 1.0) -> float:
    return compute_weight_detailed(question, boost).w_final


def compute_weight_detailed(question: str, boost: float = 1.0) -> WeightBreakdown:
    sw = s_word(question)
    ss = s_syntax(question)
    st = s_topic(question)
    sf = s_french(question)
    dynamic = 1.0 + sw + 0.5 * ss + 0.5 * st + sf
    return WeightBreakdown(
        s_word=round(sw, 3),
        s_syntax=round(ss, 3),
        s_topic=round(st, 3),
        s_french=round(sf, 3),
        w_dynamic=round(dynamic, 3),
        w_final=round(dynamic * boost, 3),
    )

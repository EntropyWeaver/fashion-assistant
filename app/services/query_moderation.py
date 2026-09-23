"""Local, multilingual moderation of fashion queries before retrieval or API use.

Detoxify classifies comments, not fashion requests. Its toxicity score alone
can be very high for harsh criticism of an item, so a score is only one signal.
The thresholds below are starting values, not calibrated safety guarantees.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Mapping, Protocol


class ModerationUnavailable(RuntimeError):
    """A query could not be assessed; do not continue to retrieval or the API."""


@dataclass(frozen=True)
class ModerationDecision:
    blocked: bool
    reason: str | None = None


class Predictor(Protocol):
    def predict(self, text: list[str]) -> Mapping[str, object]: ...


_lock = Lock()
_CLAUSE_BREAK = re.compile(r"(?<=[.!?;])\s+|,\s*")
_FASHION_WORD = re.compile(
    r"\b(?:ropa|vestid\w*|prenda\w*|camiset\w*|falda\w*|blusa\w*|"
    r"pantalon\w*|chaqueta\w*|tejido\w*|talla\w*|moda|combina\w*|"
    r"top|outfit\w*|dress\w*|jacket\w*|shirt\w*|skirt\w*|jeans|"
    r"blouse\w*|shoe\w*|boot\w*|clothes|clothing|fabric|style|fashion)\b"
)

# Strong category evidence; sexual vocabulary and obscenity alone are not
# abusive in a fashion request. Threat scores are supplemented below because
# the multilingual model under-scores some direct Spanish threats.
_CATEGORY_LIMITS = {
    "insult": 0.75,
    "identity_attack": 0.75,
    "threat": 0.45,
    "severe_toxicity": 0.70,
}


@lru_cache(maxsize=1)
def _get_predictor() -> Predictor:
    from detoxify import Detoxify

    return Detoxify("multilingual", device="cpu")


def _fashion_context(text: str) -> bool:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return bool(_FASHION_WORD.search(normalized))


def _score(scores: Mapping[str, object], name: str, index: int) -> float:
    value = float(scores[name][index])  # type: ignore[index]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid moderation score for {name}")
    return value


def _decision_for_clause(text: str, scores: Mapping[str, object], index: int) -> ModerationDecision:
    toxicity = _score(scores, "toxicity", index)
    for name, limit in _CATEGORY_LIMITS.items():
        if _score(scores, name, index) >= limit:
            return ModerationDecision(True, name)
    if toxicity >= 0.98 and _score(scores, "threat", index) >= 0.05:
        return ModerationDecision(True, "threat")
    if toxicity >= 0.98 and not _fashion_context(text):
        return ModerationDecision(True, "toxicity")
    return ModerationDecision(False)


def moderate_query(text: str, predictor: Predictor | None = None) -> ModerationDecision:
    """Return a decision, or fail closed if the local classifier cannot run.

    The supplied predictor exists for deterministic tests. In production the
    multilingual model is loaded once and inference is serialized on CPU.
    """
    clauses = [part.strip() for part in _CLAUSE_BREAK.split(text) if part.strip()]
    if not clauses:
        return ModerationDecision(False)
    try:
        if predictor is None:
            with _lock:
                scores = _get_predictor().predict(clauses)
        else:
            scores = predictor.predict(clauses)
        for index, clause in enumerate(clauses):
            decision = _decision_for_clause(clause, scores, index)
            if decision.blocked:
                return decision
        return ModerationDecision(False)
    except Exception as exc:
        raise ModerationUnavailable("Could not assess the user query") from exc

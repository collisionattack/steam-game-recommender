"""top-k ranking metrics. recs = ranked items of one player (train items removed), truth = their test items."""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set


def precision_at_k(recs: Sequence, truth: Set, k: int) -> float:
    """denominator is always k, even if recs is shorter."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for r in recs[:k] if r in truth) / k


def recall_at_k(recs: Sequence, truth: Set, k: int) -> float:
    if not truth:
        return 0.0
    return sum(1 for r in recs[:k] if r in truth) / len(truth)


def hit_rate_at_k(recs: Sequence, truth: Set, k: int) -> float:
    return 1.0 if any(r in truth for r in recs[:k]) else 0.0


def ndcg_at_k(recs: Sequence, truth: Set, k: int) -> float:
    """binary relevance; the ideal list has min(len(truth), k) hits."""
    if not truth:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(recs[:k]) if r in truth)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / ideal


def coverage(all_recs: Iterable[Sequence], k: int, n_catalog: int) -> float:
    seen = set()
    for recs in all_recs:
        seen.update(recs[:k])
    return len(seen) / n_catalog


def novelty(
    all_recs: Iterable[Sequence], k: int, item_pop: Dict, n_players: int
) -> float:
    """mean -log2(p(item)), p = share of players who have the item. higher = less popular."""
    total, count = 0.0, 0
    for recs in all_recs:
        for r in recs[:k]:
            pop = item_pop.get(r)
            if pop:
                total += -math.log2(pop / n_players)
                count += 1
    return total / count if count else 0.0


def evaluate(
    recs: Dict,
    truth: Dict,
    k: int = 10,
    n_catalog: Optional[int] = None,
    item_pop: Optional[Dict] = None,
    n_players: Optional[int] = None,
) -> Dict[str, float]:
    """mean over the players in truth; a player missing from recs counts as all misses."""
    players: List = list(truth)
    if not players:
        raise ValueError("truth is empty")
    out = {"n_players": len(players)}
    for name, fn in (
        ("precision", precision_at_k),
        ("recall", recall_at_k),
        ("hit_rate", hit_rate_at_k),
        ("ndcg", ndcg_at_k),
    ):
        out[f"{name}@{k}"] = sum(fn(recs.get(p, []), truth[p], k) for p in players) / len(players)
    if n_catalog:
        out[f"coverage@{k}"] = coverage((recs.get(p, []) for p in players), k, n_catalog)
    if item_pop is not None and n_players:
        out[f"novelty@{k}"] = novelty((recs.get(p, []) for p in players), k, item_pop, n_players)
    return out

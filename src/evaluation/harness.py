from typing import Callable

import numpy as np

from src.data.load import SplitData
from src.evaluation.metrics import evaluate, ndcg_at_k, recall_at_k


def evaluate_model(
    recommend: Callable[[np.ndarray, int], np.ndarray],
    data: SplitData,
    k: int = 10,
    batch: int = 2000,
    users: np.ndarray | None = None,
    per_user: bool = False,
) -> dict:
    """recommend(users, n) must return [len(users), n] item indices without the user's train items."""
    users = data.eval_users if users is None else users
    recs = {}
    for s in range(0, len(users), batch):
        chunk = users[s : s + batch]
        out = recommend(chunk, k)
        for u, r in zip(chunk, out):
            recs[int(u)] = [int(x) for x in r]
    truth = {int(u): data.truth[int(u)] for u in users}
    out = evaluate(
        recs, truth, k=k, n_catalog=data.n_items,
        item_pop=data.item_pop, n_players=data.n_users,
    )
    if per_user:  # arrays aligned with `users`, for significance tests / segment analysis
        out["_users"] = np.asarray(users)
        out["_ndcg"] = np.array([ndcg_at_k(recs[int(u)], truth[int(u)], k) for u in users])
        out["_recall"] = np.array([recall_at_k(recs[int(u)], truth[int(u)], k) for u in users])
    return out

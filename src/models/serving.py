"""recommendations for a visitor who picks a few games."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.models.foldin import fold_in


@dataclass
class Artifacts:
    item_ids: np.ndarray        # idx -> gameid
    item_factors: np.ndarray    # items x factors
    item_pop: np.ndarray        # players per item
    F: sp.csr_matrix            # items x content features (L2-normalised rows)

    def __post_init__(self):
        self.YtY = self.item_factors.T @ self.item_factors
        self.norms = np.linalg.norm(self.item_factors, axis=1) + 1e-9
        lp = np.log1p(self.item_pop.astype(np.float64))
        self.log_pop_z = ((lp - lp.mean()) / (lp.std() + 1e-9)).astype(np.float32)
        self.index = {int(g): i for i, g in enumerate(self.item_ids)}


def load_artifacts(path: str = "models/app_artifacts.npz") -> Artifacts:
    z = np.load(path)
    F = sp.csr_matrix((z["F_data"], z["F_indices"], z["F_indptr"]), shape=tuple(z["F_shape"]))
    return Artifacts(z["item_ids"], z["item_factors"], z["item_pop"], F)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def recommend_from_picks(
    art: Artifacts,
    picks: list,
    n: int = 10,
    alpha: float = 10.0,
    reg: float = 1.0,
    lam: float = 0.0,
    beta: float = 0.0,
    similarity: str = "dot",
) -> np.ndarray:
    """score = z(als) + lam*z(content) + beta*z(log popularity), picks removed. dot is what the offline eval used (item norm tracks popularity, so big hits dominate); cosine is norm-free."""
    picks = np.asarray(picks, dtype=np.int64)
    x = fold_in(art.item_factors, picks, np.full(len(picks), alpha), reg, art.YtY)
    if similarity == "cosine":
        s = _z((art.item_factors @ x) / (art.norms * (np.linalg.norm(x) + 1e-9)))
    elif similarity == "dot":
        s = _z(art.item_factors @ x)
    else:
        raise ValueError(similarity)
    if lam:
        profile = normalize(sp.csr_matrix(art.F[picks].sum(axis=0)))
        s = s + lam * _z((profile @ art.F.T).toarray().ravel())
    if beta:
        s = s + beta * art.log_pop_z
    s[picks] = -np.inf
    top = np.argpartition(-s, n)[:n]
    return top[np.argsort(-s[top])]

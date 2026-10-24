"""score = z(als) + lam*z(content) + beta*z(log popularity), z-scored per user. content = cosine between the user's feature profile and the item features."""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.models.als import ALSRecommender


def align_features(game_ids, F: sp.csr_matrix, item_ids) -> sp.csr_matrix:
    """reorder the rows of F to the model's item order."""
    pos = {g: i for i, g in enumerate(game_ids)}
    rows = np.array([pos[g] for g in item_ids])
    return F[rows].tocsr()


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


class HybridRecommender:
    def __init__(self, als: ALSRecommender, item_features: sp.csr_matrix,
                 lam: float = 0.3, beta: float = 0.5):
        self.als, self.F, self.lam, self.beta = als, item_features, lam, beta
        self.train = als.train
        self.train_binary = als.train_binary
        self.profiles = normalize(self.train @ self.F)  # users x features
        pop = np.asarray(self.train_binary.sum(axis=0)).ravel()
        lp = np.log1p(pop)
        self.log_pop_z = ((lp - lp.mean()) / (lp.std() + 1e-9)).astype(np.float32)

    def scores(self, user_idx) -> np.ndarray:
        user_idx = np.asarray(user_idx)
        m = self.als.model
        s = _z(np.asarray(m.user_factors[user_idx] @ m.item_factors.T))
        if self.lam:
            s += self.lam * _z((self.profiles[user_idx] @ self.F.T).toarray())
        if self.beta:
            s += self.beta * self.log_pop_z[None, :]
        return s

    def recommend(self, user_idx, n: int = 10) -> np.ndarray:
        user_idx = np.asarray(user_idx)
        s = self.scores(user_idx)
        for row, u in enumerate(user_idx):
            seen = self.train_binary.indices[
                self.train_binary.indptr[u] : self.train_binary.indptr[u + 1]
            ]
            s[row, seen] = -np.inf
        top = np.argpartition(-s, n, axis=1)[:, :n]
        order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
        return np.take_along_axis(top, order, axis=1)

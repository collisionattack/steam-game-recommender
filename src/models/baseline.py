"""popularity baseline."""
import numpy as np
import scipy.sparse as sp


class PopularityRecommender:
    """ranks by scores, default = number of train players per item."""

    def fit(self, train_binary: sp.csr_matrix, scores: np.ndarray | None = None):
        self.train_binary = train_binary
        self.pop = (
            np.asarray(train_binary.sum(axis=0)).ravel() if scores is None else np.asarray(scores)
        )
        self.order = np.argsort(-self.pop, kind="stable")
        return self

    def recommend(self, user_idx, n: int = 10) -> np.ndarray:
        out = np.empty((len(user_idx), n), dtype=np.int64)
        seen = np.zeros(self.train_binary.shape[1], dtype=bool)
        for row, u in enumerate(user_idx):
            idx = self.train_binary.indices[
                self.train_binary.indptr[u] : self.train_binary.indptr[u + 1]
            ]
            seen[idx] = True
            out[row] = self.order[~seen[self.order]][:n]
            seen[idx] = False
        return out

"""iALS via the implicit library."""
import numpy as np
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares


class ALSRecommender:
    def __init__(self, factors=64, regularization=0.1, alpha=1.0, iterations=15, seed=42):
        self.params = dict(
            factors=factors, regularization=regularization, alpha=alpha,
            iterations=iterations, random_state=seed, calculate_training_loss=False,
        )

    def fit(self, train: sp.csr_matrix, train_binary: sp.csr_matrix | None = None):
        self.model = AlternatingLeastSquares(**self.params)
        self.model.fit(train, show_progress=False)
        self.train = train
        self.train_binary = train_binary if train_binary is not None else train
        return self

    def scores(self, user_idx) -> np.ndarray:
        m = self.model
        return np.asarray(m.user_factors[np.asarray(user_idx)] @ m.item_factors.T)

    def recommend(self, user_idx, n: int = 10) -> np.ndarray:
        """top-n unseen items; seen comes from train_binary, so zero-weight rows count as seen too."""
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

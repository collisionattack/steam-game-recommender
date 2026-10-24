"""user vector for someone who is not in the training matrix: the same least-squares step iALS uses (Hu, Koren, Volinsky 2008). tests compare it with implicit's recalculate_user."""
import numpy as np


def fold_in(item_factors: np.ndarray, items, conf, reg: float, YtY: np.ndarray | None = None) -> np.ndarray:
    Y = item_factors
    if YtY is None:
        YtY = Y.T @ Y
    items = np.asarray(items, dtype=np.int64)
    c = np.asarray(conf, dtype=np.float64)
    Yi = Y[items].astype(np.float64)
    A = YtY.astype(np.float64) + (Yi * (c - 1.0)[:, None]).T @ Yi + reg * np.eye(Y.shape[1])
    b = (Yi * c[:, None]).sum(axis=0)
    return np.linalg.solve(A, b).astype(np.float32)

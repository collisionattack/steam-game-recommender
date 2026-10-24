import numpy as np
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares

from src.data.load import library_weight
from src.models.foldin import fold_in


def test_fold_in_matches_implicit():
    rng = np.random.default_rng(0)
    R = sp.random(200, 80, density=0.1, random_state=1, format="csr")
    R.data[:] = rng.integers(1, 6, R.nnz)
    alpha, reg = 10.0, 0.1
    m = AlternatingLeastSquares(factors=8, regularization=reg, alpha=alpha, iterations=4,
                                random_state=0, calculate_training_loss=False)
    m.fit(R, show_progress=False)
    row = R[7]
    ours = fold_in(m.item_factors, row.indices, row.data * alpha, reg=reg)
    theirs = np.asarray(m.recalculate_user(0, row))
    assert np.allclose(ours, theirs, atol=1e-4)


def test_library_weight_modes():
    n = np.array([1, 100, 1000, 10000])
    assert np.all(library_weight(n, "none") == 1)
    w = library_weight(n, "log")
    assert w[0] == 1 and w[1] == 1 and w[2] < 1 and w[3] < w[2]
    assert list(library_weight(n, "drop")) == [1, 1, 1, 0]


def test_recommend_from_picks_excludes_picks_and_returns_n():
    from src.models.serving import Artifacts, recommend_from_picks

    rng = np.random.default_rng(1)
    n_items, f = 50, 6
    Y = rng.normal(size=(n_items, f)).astype(np.float32)
    F = sp.csr_matrix(rng.random((n_items, 9)).astype(np.float32))
    art = Artifacts(np.arange(1000, 1000 + n_items), Y, rng.integers(1, 100, n_items), F)
    picks = [3, 7, 11]
    for lam, beta in [(0, 0), (0.5, 1.0)]:
        rec = recommend_from_picks(art, picks, n=10, lam=lam, beta=beta)
        assert len(rec) == 10 and len(set(rec)) == 10
        assert not set(rec) & set(picks)

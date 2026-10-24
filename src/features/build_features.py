"""content features: genres, developers, publishers, release year. rare developers and publishers are dropped."""
import ast
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

GAMES_CLEAN = "data/processed/games_clean.parquet"


def _parse_list(s) -> list:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    try:
        v = ast.literal_eval(s)
        return [str(x).strip() for x in v] if isinstance(v, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _year_bucket(d) -> str:
    if pd.isna(d):
        return "year:unknown"
    y = d.year
    return "year:<2008" if y < 2008 else f"year:{y}"


def build_game_features(
    games_clean: str = GAMES_CLEAN,
    min_dev: int = 3,
    min_pub: int = 3,
    w_genre: float = 1.0,
    w_dev: float = 1.0,
    w_pub: float = 0.5,
    w_year: float = 0.5,
):
    """returns (game_ids, feature_names, F); F is CSR with L2-normalised rows."""
    g = pd.read_parquet(games_clean)
    genres = g["genres"].map(_parse_list)
    devs = g["developers"].map(_parse_list)
    pubs = g["publishers"].map(_parse_list)
    years = g["release_date"].map(_year_bucket)

    dev_n = Counter(x for l in devs for x in set(l))
    pub_n = Counter(x for l in pubs for x in set(l))

    names, index = [], {}

    def col(name):
        if name not in index:
            index[name] = len(names)
            names.append(name)
        return index[name]

    rows, cols, vals = [], [], []
    for r in range(len(g)):
        feats = [(f"genre:{x}", w_genre) for x in genres.iat[r]]
        feats += [(f"dev:{x}", w_dev) for x in devs.iat[r] if dev_n[x] >= min_dev]
        feats += [(f"pub:{x}", w_pub) for x in pubs.iat[r] if pub_n[x] >= min_pub]
        feats.append((years.iat[r], w_year))
        for n, w in feats:
            rows.append(r)
            cols.append(col(n))
            vals.append(w)
    F = sp.csr_matrix((vals, (rows, cols)), shape=(len(g), len(names)), dtype=np.float32)
    return g["gameid"].to_numpy(), names, normalize(F)

"""raw csv -> interactions table -> sparse matrices for the models."""
from dataclasses import dataclass
from typing import Dict

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = "data/raw/steam"
GAMES_CLEAN = "data/processed/games_clean.parquet"
INTERACTIONS = "data/interim/interactions.parquet"
SPLIT_DIR = "data/interim"


def build_interactions(
    raw_dir: str = RAW,
    games_clean: str = GAMES_CLEAN,
    out_path: str = INTERACTIONS,
) -> None:
    """library union reviews, one row per (player, game), only games in games_clean. posted is NULL for library-only rows."""
    con = duckdb.connect()
    csv = lambda t: f"read_csv('{raw_dir}/{t}.csv', all_varchar=false, sample_size=-1)"
    con.execute(
        f"""
        COPY (
          WITH lib AS (
            SELECT playerid, unnest(CAST(library AS BIGINT[])) AS gameid
            FROM {csv('purchased_games')} WHERE library IS NOT NULL
          ),
          rev AS (SELECT playerid, gameid, posted FROM {csv('reviews')}),
          gc AS (SELECT gameid FROM read_parquet('{games_clean}'))
          SELECT
            COALESCE(l.playerid, r.playerid) AS playerid,
            COALESCE(l.gameid,   r.gameid)   AS gameid,
            (l.gameid IS NOT NULL)           AS in_library,
            (r.gameid IS NOT NULL)           AS reviewed,
            r.posted                         AS posted
          FROM (SELECT DISTINCT playerid, gameid FROM lib) l
          FULL OUTER JOIN rev r USING (playerid, gameid)
          SEMI JOIN gc ON gc.gameid = COALESCE(l.gameid, r.gameid)
        ) TO '{out_path}' (FORMAT parquet)
        """
    )


def recent_review_counts(split_dir, item_ids, cut_date: str = "2024-01-01", days: int = 365) -> np.ndarray:
    """review count per item in the `days` before cut_date, from train only."""
    con = duckdb.connect()
    r = con.sql(
        f"""SELECT gameid, COUNT(*) AS n FROM read_parquet('{split_dir}/train.parquet')
            WHERE reviewed AND posted >= DATE '{cut_date}' - INTERVAL {int(days)} DAY
            GROUP BY gameid"""
    ).df()
    pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    r = r[r["gameid"].isin(pos.index)]
    out = np.zeros(len(item_ids))
    out[pos.loc[r["gameid"]].to_numpy()] = r["n"].to_numpy()
    return out


@dataclass
class SplitData:
    user_ids: np.ndarray            # idx -> playerid
    item_ids: np.ndarray            # idx -> gameid
    train: sp.csr_matrix            # users x items, confidence weights
    train_binary: sp.csr_matrix     # users x items, 0/1 (for "already seen" masks)
    truth: Dict[int, set]           # eval user idx -> set of test item idx
    eval_users: np.ndarray          # eval user idx array
    item_pop: Dict[int, int]        # item idx -> number of train players

    @property
    def n_users(self) -> int:
        return self.train.shape[0]

    @property
    def n_items(self) -> int:
        return self.train.shape[1]


def library_weight(n_lib: np.ndarray, mode: str = "log", cap: int = 100, drop_above: int = 1000) -> np.ndarray:
    """multiplier for big libraries (none / log / drop). only 'none' ended up being used."""
    n = np.asarray(n_lib, dtype=float)
    if mode == "none":
        return np.ones_like(n)
    if mode == "log":
        return np.minimum(1.0, np.log1p(cap) / np.log1p(np.maximum(n, 1)))
    if mode == "drop":
        return (n <= drop_above).astype(float)
    raise ValueError(f"unknown mode {mode}")


def load_split_data(
    split_dir: str = SPLIT_DIR,
    weighting: str = "log",
    lib_weight: float = 1.0,
    review_weight: float = 3.0,
) -> SplitData:
    """confidence = (lib_weight*in_library + review_weight*reviewed) * library_weight(user)."""
    con = duckdb.connect()
    tr = con.sql(
        f"SELECT playerid, gameid, in_library::INT AS lib, reviewed::INT AS rev "
        f"FROM read_parquet('{split_dir}/train.parquet')"
    ).df()
    te = con.sql(f"SELECT playerid, gameid FROM read_parquet('{split_dir}/test.parquet')").df()
    ev = con.sql(f"SELECT playerid FROM read_parquet('{split_dir}/eval_players.parquet')").df()
    return _assemble(tr, te, ev, weighting, lib_weight, review_weight)


def load_serving_data(
    interactions_path: str = INTERACTIONS,
    min_item: int = 10,
    weighting: str = "log",
    lib_weight: float = 1.0,
    review_weight: float = 3.0,
) -> SplitData:
    """same matrices from all interactions, no time cut, for the app model."""
    con = duckdb.connect()
    tr = con.sql(
        f"""
        SELECT playerid, gameid, in_library::INT AS lib, reviewed::INT AS rev
        FROM read_parquet('{interactions_path}')
        SEMI JOIN (SELECT gameid FROM read_parquet('{interactions_path}')
                   GROUP BY gameid HAVING COUNT(*) >= {int(min_item)}) USING (gameid)
        """
    ).df()
    empty = pd.DataFrame({"playerid": pd.Series(dtype="int64"), "gameid": pd.Series(dtype="int64")})
    return _assemble(tr, empty, empty[["playerid"]], weighting, lib_weight, review_weight)


def _assemble(tr, te, ev, weighting, lib_weight, review_weight) -> SplitData:
    u_codes, user_ids = pd.factorize(tr["playerid"], sort=True)
    i_codes, item_ids = pd.factorize(tr["gameid"], sort=True)
    n_users, n_items = len(user_ids), len(item_ids)

    n_lib = np.bincount(u_codes, weights=tr["lib"].to_numpy(), minlength=n_users)
    mult = library_weight(n_lib, weighting)[u_codes]
    vals = lib_weight * tr["lib"].to_numpy() * mult + review_weight * tr["rev"].to_numpy()
    keep = vals > 0
    train = sp.csr_matrix(
        (vals[keep], (u_codes[keep], i_codes[keep])), shape=(n_users, n_items)
    )
    # every train interaction as 0/1, zero-weight ones included: "already seen"
    train_binary = sp.csr_matrix(
        (np.ones(len(u_codes), dtype=np.float32), (u_codes, i_codes)), shape=(n_users, n_items)
    )

    user_index = pd.Series(np.arange(n_users), index=user_ids)
    item_index = pd.Series(np.arange(n_items), index=item_ids)
    te = te[te["playerid"].isin(user_index.index) & te["gameid"].isin(item_index.index)]
    te_u = user_index.loc[te["playerid"]].to_numpy()
    te_i = item_index.loc[te["gameid"]].to_numpy()
    ev_ids = ev["playerid"][ev["playerid"].isin(user_index.index)]
    eval_set = set(user_index.loc[ev_ids].to_numpy().tolist())
    truth: Dict[int, set] = {}
    for u, i in zip(te_u, te_i):
        if int(u) in eval_set:
            truth.setdefault(int(u), set()).add(int(i))

    pop = np.bincount(i_codes, minlength=n_items)
    return SplitData(
        user_ids=np.asarray(user_ids),
        item_ids=np.asarray(item_ids),
        train=train,
        train_binary=train_binary,
        truth=truth,
        eval_users=np.array(sorted(truth), dtype=np.int64),
        item_pop={i: int(c) for i, c in enumerate(pop)},
    )

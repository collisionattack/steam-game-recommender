"""cold start: test reviews of games with < 10 train interactions, which iALS can't score. content profile vs random and newest-first."""
import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src.data.load import SPLIT_DIR, load_split_data
from src.evaluation.metrics import evaluate
from src.evaluation.split import make_split
from src.features.build_features import build_game_features
from src.models.config import load_config
from src.models.hybrid import align_features

K = 10
cfg = load_config()
con = duckdb.connect()
st = make_split(con, out_dir=None)
data = load_split_data(SPLIT_DIR, **cfg["weights"])
gid, _, F = build_game_features()
Fi = align_features(gid, F, data.item_ids)
profiles = normalize(data.train @ Fi)

games = pd.read_parquet("data/processed/games_clean.parquet")[["gameid", "release_date"]]
cf = set(map(int, data.item_ids))
cand = games[~games["gameid"].isin(cf) & (games["release_date"] <= pd.Timestamp("2025-01-10").date())].reset_index(drop=True)
cand_pos = {int(g): i for i, g in enumerate(cand["gameid"])}
frow = pd.Series(np.arange(len(gid)), index=gid)
Fc = F[frow.loc[cand["gameid"]].to_numpy()].tocsr()
days = (pd.to_datetime(cand["release_date"]) - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(float)
recency_z = (days - days.mean()) / days.std()

cold = con.sql(
    """
    SELECT t.playerid, t.gameid FROM test_pairs t
    ANTI JOIN cf_items USING (gameid)
    """
).df()
cold = cold[cold["gameid"].isin(cand_pos)]
rev2 = con.sql("SELECT playerid FROM train WHERE reviewed GROUP BY playerid HAVING COUNT(*) >= 2").df()
user_pos = pd.Series(np.arange(data.n_users), index=data.user_ids)
cold = cold[cold["playerid"].isin(rev2["playerid"]) & cold["playerid"].isin(user_pos.index)]
truth = {}
for p, g in zip(cold["playerid"], cold["gameid"]):
    truth.setdefault(int(user_pos.loc[p]), set()).add(cand_pos[int(g)])
users = np.array(sorted(truth))
print(f"cold players {len(users)}, cold test pairs {len(cold)}, candidates {len(cand)}", flush=True)

# items a player already owned/reviewed before the cut among the candidates (removed from train by the item filter)
seen_df = con.sql(
    "SELECT playerid, gameid FROM train_raw"
).df()
seen_df = seen_df[seen_df["gameid"].isin(cand_pos) & seen_df["playerid"].isin(user_pos.index)]
seen = {}
for p, g in zip(seen_df["playerid"], seen_df["gameid"]):
    u = int(user_pos.loc[p])
    if u in truth:
        seen.setdefault(u, []).append(cand_pos[int(g)])


def rank(score_fn):
    recs = {}
    for s in range(0, len(users), 1000):
        chunk = users[s : s + 1000]
        sc = score_fn(chunk)
        for r, u in enumerate(chunk):
            if u in seen:
                sc[r, seen[u]] = -np.inf
        top = np.argpartition(-sc, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(sc, top, 1), 1)
        for r, u in enumerate(chunk):
            recs[int(u)] = [int(x) for x in np.take_along_axis(top, order, 1)[r]]
    return recs


z = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
rng = np.random.default_rng(0)
methods = {
    "random": lambda c: rng.random((len(c), len(cand))),
    "newest first": lambda c: np.tile(days, (len(c), 1)),
    "content profile": lambda c: (profiles[c] @ Fc.T).toarray(),
    "content + recency": lambda c: z((profiles[c] @ Fc.T).toarray()) + recency_z[None, :],
}
rows = []
for name, fn in methods.items():
    r = evaluate(rank(fn), truth, k=K, n_catalog=len(cand))
    rows.append((name, r[f"ndcg@{K}"], r[f"recall@{K}"], r[f"hit_rate@{K}"], r[f"coverage@{K}"]))
    print(name, [round(x, 4) for x in rows[-1][1:]], flush=True)

lines = [
    "# Cold-start check (games with < 10 train interactions)", "",
    f"Players: {len(users):,}; cold test reviews: {len(cold):,}; candidate games: {len(cand):,}", "",
    "| method | NDCG@10 | Recall@10 | Hit@10 | Coverage@10 |", "|---|---|---|---|---|",
] + [f"| {n} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.3f} |" for n, a, b, c, d in rows]
open("reports/coldstart.md", "w").write("\n".join(lines) + "\n")

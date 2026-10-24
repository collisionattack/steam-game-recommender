"""rerun the final config on other cut dates, no re-tuning. appends to reports/cuts.csv."""
import csv
import os

import duckdb

from src.data.load import load_split_data, recent_review_counts
from src.evaluation.harness import evaluate_model
from src.evaluation.split import make_split
from src.features.build_features import build_game_features
from src.models.als import ALSRecommender
from src.models.baseline import PopularityRecommender
from src.models.config import load_config
from src.models.hybrid import HybridRecommender, align_features

WINDOWS = [("2022-01-01", "2023-01-01"), ("2024-07-01", None), ("2023-07-01", None)]
OUT = "reports/cuts.csv"
cfg = load_config()
w, a, h = cfg["weights"], cfg["als"], cfg["hybrid"]
gid, _, F = build_game_features()

for cut, end in WINDOWS:
    d_dir = f"data/interim/alt_{cut}"
    os.makedirs(d_dir, exist_ok=True)
    st = make_split(duckdb.connect(), cut_date=cut, end_date=end, out_dir=d_dir)
    data = load_split_data(d_dir, **w)
    Fi = align_features(gid, F, data.item_ids)
    recent = recent_review_counts(d_dir, data.item_ids, cut, 365)
    als = ALSRecommender(a["factors"], a["regularization"], a["alpha"], a["iterations"]).fit(data.train, data.train_binary)
    models = {
        "popularity (all time)": PopularityRecommender().fit(data.train_binary),
        "popularity (last 365d)": PopularityRecommender().fit(data.train_binary, recent),
        "iALS": als,
        "hybrid": HybridRecommender(als, Fi, lam=h["lam"], beta=h["beta"]),
    }
    for name, m in models.items():
        r = evaluate_model(m.recommend, data, k=10)
        row = dict(cut=cut, end=end or "", model=name, n_eval=st["n_eval_players"], n_test=st["n_test"],
                   ndcg=round(r["ndcg@10"], 4), recall=round(r["recall@10"], 4),
                   hit=round(r["hit_rate@10"], 4), coverage=round(r["coverage@10"], 4))
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        print(row, flush=True)

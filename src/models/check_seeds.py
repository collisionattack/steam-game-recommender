"""rerun iALS and hybrid with other seeds on the 2024 test."""
import numpy as np

from src.data.load import SPLIT_DIR, load_split_data
from src.evaluation.harness import evaluate_model
from src.features.build_features import build_game_features
from src.models.als import ALSRecommender
from src.models.config import load_config
from src.models.hybrid import HybridRecommender, align_features

cfg = load_config()
w, a, h = cfg["weights"], cfg["als"], cfg["hybrid"]
data = load_split_data(SPLIT_DIR, **w)
gid, _, F = build_game_features()
Fi = align_features(gid, F, data.item_ids)
rows = []
for seed in (1, 2, 3, 4, 5):
    als = ALSRecommender(a["factors"], a["regularization"], a["alpha"], a["iterations"], seed=seed).fit(data.train, data.train_binary)
    r1 = evaluate_model(als.recommend, data, k=10)
    r2 = evaluate_model(HybridRecommender(als, Fi, lam=h["lam"], beta=h["beta"]).recommend, data, k=10)
    rows.append((r1["ndcg@10"], r2["ndcg@10"]))
    print(f"seed {seed}: iALS {r1['ndcg@10']:.4f}  hybrid {r2['ndcg@10']:.4f}", flush=True)
x = np.array(rows)
print(f"iALS   mean {x[:,0].mean():.4f} sd {x[:,0].std(ddof=1):.4f}")
print(f"hybrid mean {x[:,1].mean():.4f} sd {x[:,1].std(ddof=1):.4f}")
print(f"hybrid - iALS per seed: {np.round(x[:,1]-x[:,0],4).tolist()}")

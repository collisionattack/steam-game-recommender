"""grid search for iALS on the 2023 validation split, never touches 2024. results go to reports/tuning_als.csv."""
import csv
import itertools
import os
import sys
import time

from src.data.load import load_split_data
from src.evaluation.harness import evaluate_model
from src.models.als import ALSRecommender
from src.models.baseline import PopularityRecommender

VAL = "data/interim/val"
OUT = "reports/tuning_als.csv"
FIELDS = ["stage", "weighting", "review_weight", "factors", "reg", "iters", "alpha",
          "ndcg@10", "recall@10", "hit_rate@10", "precision@10", "coverage@10", "novelty@10", "secs"]


def log(row):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    print(row, flush=True)


def run(stage, weighting, review_weight, factors, reg, iters=10, alpha=1.0):
    d = load_split_data(VAL, weighting=weighting, review_weight=review_weight)
    t = time.time()
    m = ALSRecommender(factors=factors, regularization=reg, iterations=iters, alpha=alpha).fit(d.train, d.train_binary)
    r = evaluate_model(m.recommend, d, k=10)
    log(dict(stage=stage, weighting=weighting, review_weight=review_weight, factors=factors,
             reg=reg, iters=iters, alpha=alpha, secs=round(time.time() - t), **{k: r[k] for k in FIELDS if k in r}))
    return r["ndcg@10"]


def stage4():
    """small models, based on stages 1-3: alpha 10, review weight 1, no library weighting, smaller is better."""
    best, cfg = -1, None
    for f, reg in itertools.product([8, 16, 32], [1.0, 3.0, 10.0]):
        s = run("4-small", "none", 1.0, f, reg, alpha=10.0)
        if s > best:
            best, cfg = s, (f, reg)
    for a in [3.0, 20.0]:
        run("4-alpha", "none", 1.0, cfg[0], cfg[1], alpha=a)
    for it in [5, 20]:
        run("4-iters", "none", 1.0, cfg[0], cfg[1], iters=it, alpha=10.0)


if __name__ == "__main__" and "stage4" in sys.argv:
    stage4()
    sys.exit(0)


if __name__ == "__main__":
    d = load_split_data(VAL)
    r = evaluate_model(PopularityRecommender().fit(d.train_binary).recommend, d, k=10)
    log(dict(stage="baseline", weighting="-", review_weight=0, factors=0, reg=0, iters=0, alpha=0, secs=0,
             **{k: r[k] for k in FIELDS if k in r}))

    # Stage 1: confidence scale (alpha) x regularisation, everything else fixed
    best, cfg = -1, None
    for alpha, reg in itertools.product([1.0, 10.0, 40.0], [0.01, 0.1, 1.0]):
        s = run("1-alpha-reg", "none", 3.0, 64, reg, alpha=alpha)
        if s > best:
            best, cfg = s, dict(alpha=alpha, reg=reg)
    # Stage 2: library weighting and review weight
    wbest, wcfg = best, ("none", 3.0)
    for w in ["log", "drop"]:
        s = run("2-weighting", w, 3.0, 64, cfg["reg"], alpha=cfg["alpha"])
        if s > wbest:
            wbest, wcfg = s, (w, 3.0)
    for rw in [1.0, 10.0]:
        s = run("2-review-w", wcfg[0], rw, 64, cfg["reg"], alpha=cfg["alpha"])
        if s > wbest:
            wbest, wcfg = s, (wcfg[0], rw)
    # Stage 3: model size and training length
    for f, it in [(32, 10), (128, 10), (256, 10), (128, 20)]:
        run("3-size", wcfg[0], wcfg[1], f, cfg["reg"], iters=it, alpha=cfg["alpha"])

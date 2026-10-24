"""evaluate on the 2024 test window; with --serve also fit on all data for the app. settings from models/config.json, output in reports/ and models/."""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.load import SPLIT_DIR, load_serving_data, load_split_data, recent_review_counts
from src.evaluation.harness import evaluate_model
from src.features.build_features import _parse_list, build_game_features
from src.models.als import ALSRecommender
from src.models.baseline import PopularityRecommender
from src.models.config import load_config
from src.models.hybrid import HybridRecommender, align_features

KS = (5, 10, 20)


def paired_z(a: np.ndarray, b: np.ndarray) -> float:
    """paired z of mean(a - b) over users."""
    d = a - b
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))


def segment_table(models: dict, data, users: np.ndarray, per_user: dict) -> list:
    """ndcg@10 by how many items the player has in train."""
    tb = data.train_binary
    n_total = np.asarray(tb[users].sum(axis=1)).ravel()
    segs = {
        "total train items 1-50": n_total <= 50,
        "total train items 51-300": (n_total > 50) & (n_total <= 300),
        "total train items 301+": n_total > 300,
    }
    rows = []
    for sname, mask in segs.items():
        row = {"segment": sname, "n_players": int(mask.sum())}
        for m in models:
            row[m] = float(per_user[m]["_ndcg"][mask].mean())
        rows.append(row)
    return rows


def main(serve: bool = False) -> None:
    cfg = load_config()
    w, a, h = cfg["weights"], cfg["als"], cfg["hybrid"]
    data = load_split_data(SPLIT_DIR, **w)
    print(f"users {data.n_users}, items {data.n_items}, eval users {len(data.eval_users)}", flush=True)

    recent = recent_review_counts(SPLIT_DIR, data.item_ids, "2024-01-01", 365)
    gid, _, F = build_game_features()
    Fi = align_features(gid, F, data.item_ids)

    als = ALSRecommender(a["factors"], a["regularization"], a["alpha"], a["iterations"]).fit(
        data.train, data.train_binary
    )
    models = {
        "popularity (all time)": PopularityRecommender().fit(data.train_binary),
        "popularity (last 365d)": PopularityRecommender().fit(data.train_binary, recent),
        "iALS": als,
        "hybrid": HybridRecommender(als, Fi, lam=h["lam"], beta=h["beta"]),
    }

    results, per_user = {}, {}
    for name, m in models.items():
        results[name] = {}
        for k in KS:
            r = evaluate_model(m.recommend, data, k=k, per_user=(k == 10))
            if k == 10:
                per_user[name] = {x: r.pop(x) for x in ("_users", "_ndcg", "_recall")}
            results[name][f"@{k}"] = r
        print(name, {x: round(v, 4) for x, v in results[name]["@10"].items()}, flush=True)

    base = "popularity (all time)"
    sig = {
        n: paired_z(per_user[n]["_ndcg"], per_user[base]["_ndcg"])
        for n in models if n != base
    }
    out = {
        "config": cfg,
        "n_eval_players": int(len(data.eval_users)),
        "n_items": int(data.n_items),
        "metrics": results,
        "paired_z_ndcg10_vs_popularity": sig,
        "segments": segment_table(models, data, data.eval_users, per_user),
    }
    with open("reports/results.json", "w") as f:
        json.dump(out, f, indent=2)
    write_markdown(out)

    if serve:
        serve_artifacts(cfg)


def write_markdown(out: dict) -> None:
    lines = ["# Results on the 2024 test window", "",
             f"Evaluation players: {out['n_eval_players']:,}; candidate items: {out['n_items']:,}", "",
             "| model | NDCG@10 | Recall@10 | Hit@10 | Coverage@10 | Novelty@10 | z vs popularity |",
             "|---|---|---|---|---|---|---|"]
    for name, res in out["metrics"].items():
        r = res["@10"]
        z = out["paired_z_ndcg10_vs_popularity"].get(name)
        lines.append(
            f"| {name} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | {r['hit_rate@10']:.4f} | "
            f"{r['coverage@10']:.3f} | {r['novelty@10']:.2f} | {'' if z is None else f'{z:+.1f}'} |"
        )
    lines += ["", "## NDCG@10 by player activity", ""]
    models = [k for k in out["segments"][0] if k not in ("segment", "n_players")]
    lines.append("| segment | players | " + " | ".join(models) + " |")
    lines.append("|---|---|" + "---|" * len(models))
    for s in out["segments"]:
        lines.append(f"| {s['segment']} | {s['n_players']:,} | " + " | ".join(f"{s[m]:.4f}" for m in models) + " |")
    lines += ["", "## Metrics at other cut-offs", "", "| model | " + " | ".join(f"NDCG@{k}" for k in KS) + " | " + " | ".join(f"Recall@{k}" for k in KS) + " |", "|---|" + "---|" * (2 * len(KS))]
    for name, res in out["metrics"].items():
        lines.append(
            f"| {name} | " + " | ".join(f"{res[f'@{k}'][f'ndcg@{k}']:.4f}" for k in KS) + " | "
            + " | ".join(f"{res[f'@{k}'][f'recall@{k}']:.4f}" for k in KS) + " |"
        )
    with open("reports/results.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def serve_artifacts(cfg: dict) -> None:
    """fit on all interactions and save what the app loads."""
    w, a = cfg["weights"], cfg["als"]
    data = load_serving_data(**w)
    als = ALSRecommender(a["factors"], a["regularization"], a["alpha"], a["iterations"]).fit(
        data.train, data.train_binary
    )
    gid, _, F = build_game_features()
    Fi = align_features(gid, F, data.item_ids)
    pop = np.asarray(data.train_binary.sum(axis=0)).ravel()
    np.savez_compressed(
        "models/app_artifacts.npz",
        item_ids=data.item_ids,
        item_factors=np.asarray(als.model.item_factors, dtype=np.float32),
        item_pop=pop.astype(np.int32),
        F_data=Fi.data, F_indices=Fi.indices, F_indptr=Fi.indptr, F_shape=np.array(Fi.shape),
    )
    # slim catalog for the app (titles, genres, developers, year)
    cat = pd.read_parquet("data/processed/games_clean.parquet")
    parse = lambda col: cat[col].map(_parse_list)
    out = pd.DataFrame({
        "gameid": cat["gameid"], "title": cat["title"],
        "genres": parse("genres"), "developers": parse("developers"),
        "year": cat["release_date"].map(lambda d: None if pd.isna(d) else int(d.year)),
    })
    out = out[out["gameid"].isin(set(map(int, data.item_ids)))]
    out.to_parquet("models/app_catalog.parquet")
    print("saved models/app_artifacts.npz and models/app_catalog.parquet", flush=True)


if __name__ == "__main__":
    main(serve="--serve" in sys.argv)

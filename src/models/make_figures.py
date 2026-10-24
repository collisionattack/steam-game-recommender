"""figures for the README."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9c9b95"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_models():
    res = json.load(open("reports/results.json"))["metrics"]
    names = ["popularity (all time)", "popularity (last 365d)", "iALS", "hybrid"]
    labels = ["Popularity\n(all time)", "Popularity\n(last 365 days)", "iALS", "Hybrid"]
    colors = [GRAY, GRAY, BLUE, ORANGE]
    vals = [res[n]["@10"]["ndcg@10"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160, facecolor=SURFACE)
    style(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.3f}", ha="center", color=INK, fontsize=11)
    ax.set_ylabel("NDCG@10", color=MUTED)
    ax.set_title("Held-out 2024 reviews: personalised models beat popularity", color=INK, fontsize=12, loc="left")
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig("reports/figures/model_comparison.png", facecolor=SURFACE)


def fig_tradeoff():
    d = pd.read_csv("reports/tradeoff_cosine.csv")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160, facecolor=SURFACE)
    style(ax)
    ax.plot(d["coverage@10"] * 100, d["ndcg@10"], color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    offsets = {4: (4, -16), 3: (-4, 10)}  # keep the two crowded points apart
    for _, r in d.iterrows():
        ax.annotate(f"β={r['beta']:g}", (r["coverage@10"] * 100, r["ndcg@10"]), textcoords="offset points",
                    xytext=offsets.get(int(r["beta"]) if r["beta"] in (3, 4) else -1, (6, 7)),
                    color=MUTED, fontsize=10)
    ax.set_xlabel("Catalogue coverage@10 (% of games ever recommended)", color=MUTED)
    ax.set_ylabel("NDCG@10", color=MUTED)
    ax.set_title("Accuracy vs. variety (validation window, cosine score + popularity weight β)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig("reports/figures/accuracy_vs_coverage.png", facecolor=SURFACE)


if __name__ == "__main__":
    fig_models()
    fig_tradeoff()

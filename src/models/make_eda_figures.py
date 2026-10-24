"""EDA figures for the README, from data/interim/interactions.parquet."""
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
con = duckdb.connect()
IX = "read_parquet('data/interim/interactions.parquet')"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"reports/figures/{name}.png", facecolor=SURFACE)
    plt.close(fig)


# 1. library size per player (log x)
n = con.sql(f"SELECT COUNT(*) AS n FROM {IX} WHERE in_library GROUP BY playerid").df()["n"].to_numpy()
fig, ax = plt.subplots(figsize=(7, 4), dpi=160, facecolor=SURFACE)
style(ax)
ax.hist(n, bins=np.logspace(0, np.log10(n.max()), 50), color=BLUE)
ax.set_xscale("log")
ax.set_xlabel("games in library (log scale)", color=MUTED)
ax.set_ylabel("players", color=MUTED)
ax.axvline(np.median(n), color=MUTED, linewidth=1, linestyle="--")
ax.text(np.median(n) * 1.1, ax.get_ylim()[1] * 0.92, f"median {int(np.median(n))}", color=MUTED, fontsize=10)
ax.set_title(f"Library size: median {int(np.median(n))}, max {int(n.max()):,} games", color=INK, fontsize=12, loc="left")
save(fig, "eda_library_size")

# 2. interactions per game, rank-frequency (log-log)
g = con.sql(f"SELECT COUNT(*) AS n FROM {IX} GROUP BY gameid ORDER BY n DESC").df()["n"].to_numpy()
fig, ax = plt.subplots(figsize=(7, 4), dpi=160, facecolor=SURFACE)
style(ax)
ax.plot(np.arange(1, len(g) + 1), g, color=BLUE, linewidth=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhline(10, color=MUTED, linewidth=1, linestyle="--")
ax.text(1.5, 11.5, "10 interactions: minimum for iALS", color=MUTED, fontsize=10)
ax.set_xlabel("games, ranked by interactions (log scale)", color=MUTED)
ax.set_ylabel("interactions per game (log scale)", color=MUTED)
ax.set_title(f"Long tail: {int((g < 10).sum()):,} of {len(g):,} games have fewer than 10", color=INK, fontsize=12, loc="left")
save(fig, "eda_long_tail")

# 3. reviews per year
y = con.sql(f"SELECT year(posted) AS yr, COUNT(*) AS n FROM {IX} WHERE reviewed GROUP BY 1 ORDER BY 1").df()
y = y[y["yr"] >= 2012]
fig, ax = plt.subplots(figsize=(7, 4), dpi=160, facecolor=SURFACE)
style(ax)
ax.bar(y["yr"].astype(str), y["n"] / 1000, color=BLUE, width=0.65)
ax.set_ylabel("reviews (thousands)", color=MUTED)
ax.set_title("Reviews per year (2025 is only the first 9 days)", color=INK, fontsize=12, loc="left")
plt.setp(ax.get_xticklabels(), rotation=45)
save(fig, "eda_reviews_per_year")

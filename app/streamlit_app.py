"""Steam game recommender demo.  Run from the project root:

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.config import load_config  # noqa: E402
from src.models.serving import load_artifacts, recommend_from_picks  # noqa: E402

st.set_page_config(page_title="Steam Game Recommender", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load():
    art = load_artifacts(str(ROOT / "models/app_artifacts.npz"))
    cat = pd.read_parquet(ROOT / "models/app_catalog.parquet").set_index("gameid")
    cfg = load_config(str(ROOT / "models/config.json"))
    return art, cat, cfg


def title(cat, gid) -> str:
    row = cat.loc[gid]
    year = f" ({row['year']})" if pd.notna(row["year"]) else ""
    return f"{row['title']}{year}"


def explain(cat, art, picks, rec_gid) -> str:
    """Short reason: shared developer or genres with the picked games."""
    r = cat.loc[rec_gid]
    g_rec = set(r["genres"])
    same_dev = [title(cat, art.item_ids[p]) for p in picks if set(cat.loc[art.item_ids[p]]["developers"]) & set(r["developers"])]
    if same_dev:
        return "Same developer as " + same_dev[0]
    shared = [g for g in sorted(g_rec) if any(g in cat.loc[art.item_ids[p]]["genres"] for p in picks)]
    return ("Shares genres: " + ", ".join(shared[:3])) if shared else "Popular with players who own similar games"


art, cat, cfg = load()
a, h = cfg["als"], cfg["hybrid"]

st.title("Steam Game Recommender")
st.caption(
    "Pick games you like and get suggestions from a model trained on 213k Steam players' "
    "libraries and reviews. Nothing you pick is stored."
)

with st.sidebar:
    st.header("Settings")
    n = st.slider("Number of recommendations", 5, 30, 10)
    beta = st.slider("Popularity bias", 0.0, 4.0, 1.0, 0.25,
                     help="0 = pure taste similarity (finds niche games, but often off-target); "
                          "higher = favours well-known games. On held-out data, about 3 is the most accurate.")
    lam = st.slider("Genre/developer similarity", 0.0, 2.0, 0.5, 0.1,
                    help="Adds a content-based term so recommendations share genres/developers with your picks.")
    compare = st.checkbox("Show most popular games for comparison", value=True)

# Pick list: the most popular 8,000 games keep the dropdown fast and relevant.
top = np.argsort(-art.item_pop)[:8000]
labels = {title(cat, art.item_ids[i]): int(i) for i in top}
# ?picks=Hades (2020)|Terraria (2011) pre-fills the picker, handy for sharing a link
preset = [x for x in st.query_params.get("picks", "").split("|") if x in labels]
chosen = st.multiselect("Games you like (type to search)", list(labels), default=preset,
                        placeholder="e.g. Portal 2, Hades, Stardew Valley")

if not chosen:
    st.info("Pick at least one game to get recommendations. 3-10 picks work best.")
    st.stop()

picks = [labels[c] for c in chosen]
rec = recommend_from_picks(art, picks, n=n, alpha=a["alpha"], reg=a["regularization"], lam=lam, beta=beta, similarity="cosine")


def table(idx, compact=False):
    rows = []
    for i in idx:
        gid = int(art.item_ids[i])
        r = cat.loc[gid]
        row = {"Game": title(cat, gid)}
        if compact:
            row["Genres"] = ", ".join(r["genres"][:3])
        else:
            row["Why"] = explain(cat, art, picks, gid)
        row["Steam"] = f"https://store.steampowered.com/app/{gid}"
        rows.append(row)
    return pd.DataFrame(rows)


cfg_cols = {"Steam": st.column_config.LinkColumn("Steam", display_text="open")}
left, right = st.columns([3, 2]) if compare else (st.container(), None)
with left:
    st.subheader("Recommended for you")
    st.dataframe(table(rec), hide_index=True, width="stretch", column_config=cfg_cols)
if compare:
    with right:
        st.subheader("Most popular (not already picked)")
        pop_idx = [i for i in np.argsort(-art.item_pop) if i not in set(picks)][:n]
        st.dataframe(table(pop_idx, compact=True), hide_index=True, width="stretch", column_config=cfg_cols)

with st.expander("How it works"):
    st.markdown(
        f"""
The model is trained on game libraries and reviews of about 213k public Steam profiles (Kaggle, Gaming Profiles 2025).
It is an iALS matrix factorisation with {a['factors']} factors. Your picks are turned into a taste vector with the same
least-squares step the model uses for its own players, then every game is scored.

The final score is the taste similarity (cosine), plus a genre/developer match, plus the popularity bias times log(popularity).
The sliders on the left change those weights.

More popularity makes the list more accurate and less varied. On held-out 2023 reviews, a popularity weight of 0 gave
NDCG@10 0.017 and recommended 22% of the catalogue; a weight of 3 gave 0.078 and 1.5% of the catalogue.

The accuracy numbers come from players with long histories. With 3 picks the model has much less to go on.
"""
    )

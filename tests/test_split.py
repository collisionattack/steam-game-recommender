import duckdb
import pandas as pd
import pytest

from src.evaluation.split import make_split


@pytest.fixture
def interactions(tmp_path):
    """Tiny hand-made interactions table.

    cut = 2024-01-01. Items: 1,2 popular (>=2 rows after masking), 3 only in test.
    Player 10: reviews 1,2 before the cut; reviews 3 after the cut -> eval player.
    Player 20: library has 1 and 2; reviews 2 after the cut (also in library!).
    Player 30: one review before the cut, one after -> only 1 train review.
    """
    rows = [
        # playerid, gameid, in_library, reviewed, posted
        (10, 1, True, True, "2020-01-01"),
        (10, 2, True, True, "2021-01-01"),
        (10, 3, False, True, "2024-03-01"),
        (20, 1, True, False, None),
        (20, 2, True, True, "2024-05-01"),
        (30, 1, False, True, "2022-01-01"),
        (30, 2, False, True, "2024-02-01"),
        (40, 1, True, False, None),
        (40, 2, True, False, None),
        (40, 3, True, False, None),
    ]
    df = pd.DataFrame(rows, columns=["playerid", "gameid", "in_library", "reviewed", "posted"])
    df["posted"] = pd.to_datetime(df["posted"]).dt.date
    path = tmp_path / "ix.parquet"
    df.to_parquet(path)
    return str(path)


def run(interactions, **kw):
    con = duckdb.connect()
    stats = make_split(con, interactions_path=interactions, out_dir=None, **kw)
    return con, stats


def test_no_pair_in_both_and_library_mask(interactions):
    con, _ = run(interactions, min_item_train=1, min_user_train_reviews=1)
    # (20, 2) is a test review AND a library row -> must be gone from train
    assert con.sql("SELECT COUNT(*) FROM train WHERE playerid=20 AND gameid=2").fetchone()[0] == 0
    assert con.sql("SELECT COUNT(*) FROM test WHERE playerid=20 AND gameid=2").fetchone()[0] == 1
    # library-only rows stay in train
    assert con.sql("SELECT COUNT(*) FROM train WHERE playerid=20 AND gameid=1").fetchone()[0] == 1


def test_time_rule(interactions):
    con, _ = run(interactions, min_item_train=1, min_user_train_reviews=1)
    assert con.sql("SELECT COUNT(*) FROM train WHERE posted >= DATE '2024-01-01'").fetchone()[0] == 0
    assert con.sql("SELECT COUNT(*) FROM test WHERE posted < DATE '2024-01-01'").fetchone()[0] == 0


def test_item_filter_drops_unseen_test_items(interactions):
    # item 3 has only one train row (player 40 library) -> removed when min_item_train=2
    con, stats = run(interactions, min_item_train=2, min_user_train_reviews=1)
    assert con.sql("SELECT COUNT(*) FROM test WHERE gameid=3").fetchone()[0] == 0
    assert stats["n_test_before_item_filter"] == 3
    assert stats["n_test"] == 2


def test_eval_players_need_train_reviews(interactions):
    con, _ = run(interactions, min_item_train=1, min_user_train_reviews=2)
    ids = {r[0] for r in con.sql("SELECT playerid FROM eval_players").fetchall()}
    assert ids == {10}  # 20 has 0 train reviews, 30 has 1
    con, _ = run(interactions, min_item_train=1, min_user_train_reviews=1)
    ids = {r[0] for r in con.sql("SELECT playerid FROM eval_players").fetchall()}
    assert ids == {10, 30}  # 20 has a test review but no train review (library does not count)


def test_end_date_removes_future_reviews(interactions):
    con, _ = run(interactions, cut_date="2023-01-01", end_date="2024-01-01",
                 min_item_train=1, min_user_train_reviews=1)
    assert con.sql("SELECT COUNT(*) FROM test").fetchone()[0] == 0  # nothing reviewed in 2023
    assert con.sql("SELECT COUNT(*) FROM train WHERE posted >= DATE '2024-01-01'").fetchone()[0] == 0

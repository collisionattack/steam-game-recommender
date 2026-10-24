"""

Input : data/interim/interactions.parquet
        (playerid, gameid, in_library, reviewed, posted)
        posted is NULL for library-only rows.
Output: data/interim/train.parquet, test.parquet, eval_players.parquet



kurallar:
- cut_date on reviews.posted: train < cut_date, test >= cut_date.
- Library has no timestamps -> library-only rows belong to train.
- Any (playerid, gameid) pair that is a TEST review must not appear in train,
  not even as a library row (the library is a Jan-2025 snapshot, 89% of
  reviewed games are also in the library -> leakage otherwise).
- CF items: games with >= min_item_train interactions in the TRAIN table.
- Eval players: have a test review AND >= min_user_train_reviews train reviews.
"""
import duckdb

INTERACTIONS = "data/interim/interactions.parquet"
OUT_DIR = "data/interim"


def build_train_test(
    con: duckdb.DuckDBPyConnection,
    interactions_path: str = INTERACTIONS,
    cut_date: str = "2024-01-01",
    min_item_train: int = 10,
    end_date: str | None = None,
) -> dict:
    """builds `train` and `test` on con and returns row counts. end_date drops reviews from that date on (2023 validation split)."""
    where = "WHERE NOT (reviewed AND posted >= CAST(? AS DATE))" if end_date else ""
    con.execute(
        f"CREATE OR REPLACE TABLE ix AS SELECT * FROM read_parquet('{interactions_path}') {where}",
        [end_date] if end_date else [],
    )

    # reviewed on/after the cut = test pair
    con.execute(
        """
        CREATE OR REPLACE TABLE test_pairs AS
        SELECT playerid, gameid, posted FROM ix
        WHERE reviewed AND posted >= CAST(? AS DATE)
        """,
        [cut_date],
    )

    # train = everything except the test pairs. a test review that is also in
    # the library is one row, so it goes too (otherwise the library leaks it).
    con.execute(
        """
        CREATE OR REPLACE TABLE train_raw AS
        SELECT * FROM ix
        ANTI JOIN test_pairs USING (playerid, gameid)
        """
    )

    # item filter, counted after masking
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cf_items AS
        SELECT gameid FROM train_raw
        GROUP BY gameid HAVING COUNT(*) >= {int(min_item_train)}
        """
    )
    con.execute(
        "CREATE OR REPLACE TABLE train AS SELECT * FROM train_raw SEMI JOIN cf_items USING (gameid)"
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE test AS
        SELECT playerid, gameid, posted FROM test_pairs
        SEMI JOIN cf_items USING (gameid)
        """
    )

    def n(q):
        return con.sql(q).fetchone()[0]

    return {
        "n_test_before_item_filter": n("SELECT COUNT(*) FROM test_pairs"),
        "n_test": n("SELECT COUNT(*) FROM test"),
        "n_train_before_item_filter": n("SELECT COUNT(*) FROM train_raw"),
        "n_train": n("SELECT COUNT(*) FROM train"),
        "n_cf_items": n("SELECT COUNT(*) FROM cf_items"),
    }


def build_eval_players(
    con: duckdb.DuckDBPyConnection,
    min_user_train_reviews: int = 2,
) -> int:
    """players with a test row and >= min_user_train_reviews train reviews. library rows don't count."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE eval_players AS
        SELECT playerid FROM train WHERE reviewed
        GROUP BY playerid HAVING COUNT(*) >= {int(min_user_train_reviews)}
        INTERSECT
        SELECT DISTINCT playerid FROM test
        """
    )
    return con.sql("SELECT COUNT(*) FROM eval_players").fetchone()[0]


def save_split(con: duckdb.DuckDBPyConnection, out_dir: str = OUT_DIR) -> None:
    for t in ("train", "test", "eval_players"):
        con.execute(f"COPY {t} TO '{out_dir}/{t}.parquet' (FORMAT parquet)")


def validate_split(con: duckdb.DuckDBPyConnection, cut_date: str = "2024-01-01") -> dict:
    """raises if train and test overlap, train has rows after cut_date, test has rows before it (or undated), or a test item is missing from train."""

    def n(q, *p):
        return con.execute(q, list(p)).fetchone()[0]

    overlap = n("SELECT COUNT(*) FROM train JOIN test USING (playerid, gameid)")
    assert overlap == 0, f"{overlap} pairs are in both train and test"

    late_train = n("SELECT COUNT(*) FROM train WHERE posted >= CAST(? AS DATE)", cut_date)
    assert late_train == 0, f"{late_train} train rows have posted >= cut_date"

    bad_test = n(
        "SELECT COUNT(*) FROM test WHERE posted IS NULL OR posted < CAST(? AS DATE)",
        cut_date,
    )
    assert bad_test == 0, f"{bad_test} test rows are before the cut or undated"

    unseen_items = n(
        "SELECT COUNT(*) FROM (SELECT DISTINCT gameid FROM test) t "
        "ANTI JOIN (SELECT DISTINCT gameid FROM train) r USING (gameid)"
    )
    assert unseen_items == 0, f"{unseen_items} test items are not in train"

    return {
        "n_train": n("SELECT COUNT(*) FROM train"),
        "n_test": n("SELECT COUNT(*) FROM test"),
        "n_eval_players": n("SELECT COUNT(*) FROM eval_players"),
        "n_items": n("SELECT COUNT(DISTINCT gameid) FROM train"),
    }


def make_split(
    con: duckdb.DuckDBPyConnection,
    cut_date: str = "2024-01-01",
    min_item_train: int = 10,
    min_user_train_reviews: int = 2,
    interactions_path: str = INTERACTIONS,
    out_dir: str | None = OUT_DIR,
    end_date: str | None = None,
) -> dict:
    stats = build_train_test(con, interactions_path, cut_date, min_item_train, end_date)
    build_eval_players(con, min_user_train_reviews)
    stats.update(validate_split(con, cut_date))
    if out_dir:
        save_split(con, out_dir)
    return stats

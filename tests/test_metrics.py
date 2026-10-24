import math

import pytest

from src.evaluation.metrics import (
    coverage,
    evaluate,
    hit_rate_at_k,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)

# Hand-worked example: ranked list a b c d e, relevant items {b, d, z}.
RECS = ["a", "b", "c", "d", "e"]
TRUTH = {"b", "d", "z"}


def test_precision():
    assert precision_at_k(RECS, TRUTH, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RECS, TRUTH, 2) == pytest.approx(1 / 2)
    # list shorter than k still divides by k
    assert precision_at_k(["b"], TRUTH, 4) == pytest.approx(1 / 4)


def test_recall():
    assert recall_at_k(RECS, TRUTH, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RECS, TRUTH, 1) == 0.0
    assert recall_at_k(RECS, set(), 5) == 0.0


def test_hit_rate():
    assert hit_rate_at_k(RECS, TRUTH, 1) == 0.0
    assert hit_rate_at_k(RECS, TRUTH, 2) == 1.0


def test_ndcg():
    # hits at ranks 2 and 4 (0-based 1, 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 relevant, k=5
    assert ndcg_at_k(RECS, TRUTH, 5) == pytest.approx(dcg / ideal)
    # perfect ranking -> 1
    assert ndcg_at_k(["b", "d", "z"], TRUTH, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], TRUTH, 2) == 0.0


def test_ndcg_ideal_capped_by_k():
    # 3 relevant but k=2: ideal list has only 2 slots
    assert ndcg_at_k(["b", "d"], TRUTH, 2) == pytest.approx(1.0)


def test_coverage():
    all_recs = [["a", "b"], ["b", "c"], ["a", "c"]]
    assert coverage(all_recs, 2, n_catalog=10) == pytest.approx(3 / 10)
    assert coverage(all_recs, 1, n_catalog=10) == pytest.approx(2 / 10)


def test_novelty():
    pop = {"a": 50, "b": 25}
    # p(a)=0.5 -> 1 bit, p(b)=0.25 -> 2 bits, mean 1.5
    assert novelty([["a", "b"]], 2, pop, n_players=100) == pytest.approx(1.5)


def test_evaluate_three_players():
    recs = {1: ["a", "b"], 2: ["x", "y"], 3: ["c", "d"]}
    truth = {1: {"a"}, 2: {"z"}, 3: {"d", "e"}}
    r = evaluate(recs, truth, k=2, n_catalog=6)
    assert r["n_players"] == 3
    assert r["precision@2"] == pytest.approx((1 / 2 + 0 + 1 / 2) / 3)
    assert r["recall@2"] == pytest.approx((1 + 0 + 1 / 2) / 3)
    assert r["hit_rate@2"] == pytest.approx(2 / 3)
    assert r["coverage@2"] == pytest.approx(6 / 6)


def test_evaluate_missing_player_counts_as_miss():
    r = evaluate({}, {1: {"a"}}, k=3)
    assert r["recall@3"] == 0.0

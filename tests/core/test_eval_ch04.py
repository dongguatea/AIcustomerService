import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import eval_ch04 as ev  # noqa: E402


def test_norm_strips_whitespace():
    assert ev._norm("每 2 周") == "每2周"
    assert ev._norm("满 99 元包邮") == "满99元包邮"


def test_recall_at():
    assert ev._recall_at(3, 5) == 1.0
    assert ev._recall_at(5, 5) == 1.0
    assert ev._recall_at(6, 5) == 0.0
    assert ev._recall_at(0, 5) == 0.0


def test_rr():
    assert ev._rr(1) == 1.0
    assert ev._rr(2) == 0.5
    assert ev._rr(0) == 0.0


def test_hit_rank_matches_section_keyword():
    hits = [
        {"section_path": "商品与购物 FAQ / 发货时效"},
        {"section_path": "商品与购物 FAQ / 运费怎么算"},
    ]
    assert ev._hit_rank(hits, ["运费怎么算"]) == 2
    assert ev._hit_rank(hits, ["不存在的章节"]) == 0


def test_coverage_mech_fraction():
    hits = [{"question": "运费", "answer": "满 99 元包邮,不满收 10 元"}]
    assert ev._coverage_mech(["满 99 元包邮", "10 元运费"], hits) == 0.5
    assert ev._coverage_mech([], hits) is None

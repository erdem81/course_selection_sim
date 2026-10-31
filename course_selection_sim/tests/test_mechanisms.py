"""基本正确性检查。两种运行方式均可：
    python tests/test_mechanisms.py        （不需要额外安装）
    python -m pytest -q                    （若已安装 pytest）
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim.population import make_instance                      # noqa: E402
from sim.mechanisms import MECHANISMS, run_mechanism, serial_dictatorship  # noqa: E402
from sim.metrics import evaluate                               # noqa: E402
from sim import paper_demo as P                                # noqa: E402

CFG = {k: v for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "default.json"),
                                       encoding="utf-8")).items() if not k.startswith("_")}


def test_constraints_and_no_wasted_seats():
    for name in MECHANISMS:
        inst = make_instance(CFG, np.random.default_rng(0))
        X, _ = run_mechanism(name, inst, np.random.default_rng(1), CFG)
        assert (X.sum(axis=1) <= inst.k).all(), name
        assert (X.sum(axis=0) <= inst.cap).all(), name
        remaining = inst.cap - X.sum(axis=0)
        for i in np.where(X.sum(axis=1) < inst.k)[0]:
            # 未选满的人，不应还存在“有空位且自己没选过”的课程
            assert not ((remaining > 0) & ~X[i]).any(), name


def test_reproducible():
    for name in MECHANISMS:
        inst = make_instance(CFG, np.random.default_rng(3))
        X1, _ = run_mechanism(name, inst, np.random.default_rng(5), CFG)
        X2, _ = run_mechanism(name, inst, np.random.default_rng(5), CFG)
        assert (X1 == X2).all(), name


def test_serial_dictatorship_order_matters():
    B = np.array([[0.9, 0.1, 0.5], [0.8, 0.7, 0.1]])
    X = serial_dictatorship(B, np.array([1, 1, 1]), 1, order=[1, 0])
    assert X[1, 0] and X[0, 2]


def test_metric_bounds():
    cfg = dict(CFG, sigma=0.0)
    inst = make_instance(cfg, np.random.default_rng(4))
    X, _ = run_mechanism("SnakeDraft", inst, np.random.default_rng(4), cfg)
    m, norm = evaluate(inst, X, cfg["k"])
    assert 0 <= m["satisfaction_mean"] <= 1 and (norm <= 1 + 1e-9).all()
    assert 0 < m["jain"] <= 1


def test_envy_definition_matches_paper():
    ua, ub = np.array([0.9, 0.2]), np.array([0.1, 0.8])
    x = np.array([True, False])                  # 物品0给a，物品1给b
    ea, eb = P.envies(x, ua, ub)
    assert math.isclose(ea, 0.2 - 0.9) and math.isclose(eb, 0.1 - 0.8)
    assert P.is_envy_free(x, ua, ub)


def test_brute_force_finds_envy_free_when_exists():
    ua, ub = P.skewed_instance(10, np.random.default_rng(0))
    assert P.is_envy_free(P.brute_force(ua, ub), ua, ub)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("通过", t.__name__)
    print(f"全部 {len(tests)} 项测试通过")

"""论文 (Li, Ling, Scarlett, Suksompong, ICML 2026) 两人模型的小规模演示实验。

Small illustrative experiments for the two-agent noisy-query model of the paper.
注意：论文是纯理论工作，没有实验；这里的结果只用于直观演示算法行为，
不能当作对 Θ̃(m^2.5/Δ^2) 界的数值验证。

- 物品 i 对 a、b 的真实效用 u_a[i], u_b[i] ∈ [0, 1]，加性。
- 每次查询物品 i 得到 (u_a[i] + N(0, σ²), u_b[i] + N(0, σ²))；每件物品查询 tau 次取平均。
- 无嫉妒(EF)：max(Envy_{a→b}, Envy_{b→a}) ≤ 0（用真实效用判断）。
"""
import time
import numpy as np


def skewed_instance(m, rng):
    """a 对所有物品的估值普遍高于 b（此时 c = 1 的朴素阈值会把大多数物品给 a）。"""
    ua = rng.uniform(0.3, 1.0, size=m)
    ub = rng.uniform(0.0, 0.7, size=m)
    return ua, ub


def envies(x_to_a, ua, ub):
    """x_to_a: bool 数组，True 表示物品给 a。返回 (Envy_{a→b}, Envy_{b→a})。"""
    s = np.where(x_to_a, 1.0, -1.0)
    envy_a = -(s * ua).sum()   # Σ_{A_b} u_a − Σ_{A_a} u_a
    envy_b = (s * ub).sum()    # Σ_{A_a} u_b − Σ_{A_b} u_b
    return envy_a, envy_b


def is_envy_free(x_to_a, ua, ub):
    ea, eb = envies(x_to_a, ua, ub)
    return max(ea, eb) <= 1e-12


def threshold_rule(va, vb, c):
    """论文式 (4)：c·v_a − v_b > 0 则给 a，否则给 b。"""
    return c * va - vb > 0


C_GRID = np.unique(np.concatenate([np.geomspace(0.05, 20.0, 400), [1.0]]))


def tuned_threshold(va, vb, grid=C_GRID):
    """在网格上选 c，使“按估计值计算的两人嫉妒中较大者”最小（平衡两人的嫉妒）。

    这是对论文中 c 选择规则（附录 C, 引理 9：只依赖估计值的平衡条件）的简化实现。
    """
    best_x, best_obj = None, np.inf
    for c in grid:
        x = threshold_rule(va, vb, c)
        obj = max(envies(x, va, vb))
        if obj < best_obj:
            best_obj, best_x = obj, x
    return best_x


def brute_force(va, vb, chunk=1 << 16):
    """论文算法 1 的第 4 步：枚举全部 2^m 种分配，选 min(a 的余量, b 的余量) 最大的。"""
    m = len(va)
    best_val, best_row = -np.inf, None
    total = 2 ** m
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)[:, None]
        S = np.where(((idx >> np.arange(m)) & 1) == 1, 1.0, -1.0)
        margin_a = S @ va           # v_a(A_a) − v_a(A_b)
        margin_b = -(S @ vb)        # v_b(A_b) − v_b(A_a)
        obj = np.minimum(margin_a, margin_b)
        r = int(np.argmax(obj))
        if obj[r] > best_val:
            best_val, best_row = obj[r], S[r]
    return best_row > 0


def optimal_gap(ua, ub):
    """真实效用下的最优负嫉妒 Δ（只对小 m 可算）：Δ = −min_A Envy(A)。"""
    x = brute_force(ua, ub)
    return -max(envies(x, ua, ub))


def random_allocation(m, rng):
    return rng.random(m) < 0.5


def success_vs_queries(m, taus, trials, sigma, rng, include_brute=False):
    """每件物品查询 tau 次时，各方法输出“真实无嫉妒”分配的比例。"""
    methods = ["threshold_c1", "threshold_tuned", "random"] + (["brute_force"] if include_brute else [])
    res = {k: [] for k in methods}
    gaps = []
    for tau in taus:
        hits = {k: 0 for k in methods}
        for _ in range(trials):
            ua, ub = skewed_instance(m, rng)
            if include_brute and tau == taus[0]:
                gaps.append(optimal_gap(ua, ub))
            va = ua + rng.normal(0, sigma / np.sqrt(tau), size=m)
            vb = ub + rng.normal(0, sigma / np.sqrt(tau), size=m)
            hits["threshold_c1"] += is_envy_free(threshold_rule(va, vb, 1.0), ua, ub)
            hits["threshold_tuned"] += is_envy_free(tuned_threshold(va, vb), ua, ub)
            hits["random"] += is_envy_free(random_allocation(m, rng), ua, ub)
            if include_brute:
                hits["brute_force"] += is_envy_free(brute_force(va, vb), ua, ub)
        for k in methods:
            res[k].append(hits[k] / trials)
    return res, (float(np.mean(gaps)) if gaps else None)


def random_allocation_fluctuation(ms, trials, rng):
    """完全随机分配（每件物品抛硬币）：嫉妒的均值≈0，但标准差随 √m 增长。

    两人都不嫉妒的概率约 12%，而不是直觉上的 1/4：a 拿到的越多，b 眼中 a 那份也越多，
    两个条件高度冲突（u ~ U(0,1) 时相关系数 0.75，正态近似 1/4 − arcsin(0.75)/2π ≈ 11.5%）。
    """
    out = {"m": [], "envy_mean": [], "envy_std": [], "p_ef": []}
    for m in ms:
        ea_all, ef = [], 0
        for _ in range(trials):
            ua = rng.uniform(0, 1, size=m)
            ub = rng.uniform(0, 1, size=m)
            x = random_allocation(m, rng)
            ea, eb = envies(x, ua, ub)
            ea_all.append(ea)
            ef += max(ea, eb) <= 0
        out["m"].append(m)
        out["envy_mean"].append(float(np.mean(ea_all)))
        out["envy_std"].append(float(np.std(ea_all)))
        out["p_ef"].append(ef / trials)
    return out


def runtime_brute_vs_threshold(ms, rng, repeats=3):
    """技术权衡：枚举法 O(2^m) vs 逐项阈值法（多项式时间）的运行时间。"""
    out = {"m": [], "brute_s": [], "threshold_s": []}
    for m in ms:
        ua, ub = skewed_instance(m, rng)
        t_b, t_t = [], []
        for _ in range(repeats):
            t0 = time.perf_counter(); brute_force(ua, ub); t_b.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); tuned_threshold(ua, ub); t_t.append(time.perf_counter() - t0)
        out["m"].append(m)
        out["brute_s"].append(float(np.median(t_b)))
        out["threshold_s"].append(float(np.median(t_t)))
    return out

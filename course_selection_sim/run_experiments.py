"""运行全部实验，输出 results/*.csv、results/summary.json 和 figures/*.png。

用法：
    python run_experiments.py                # 全部实验（约 2–4 分钟）
    python run_experiments.py --quick        # 快速检查（少量随机种子）
    python run_experiments.py --exp e1 p1    # 只跑指定实验
    python run_experiments.py --config config/my.json   # 用问卷校准后的参数

实验列表：
    e1  五种机制在默认参数下的对比（主结果）
    e2  座位紧张程度（seat_ratio）变化时的对比
    e3  偏好噪声下，“多问几次”（tau）对结果的影响 —— 对应论文的噪声查询模型
    e4  连续 8 个学期：吃亏的是不是总是同一批人
    p1  论文两人模型：阈值规则 vs 不调 c vs 枚举 vs 随机分配，成功率随查询次数的变化
    p2  论文两人模型：完全随机分配“平均公平”，但嫉妒的波动随 √m 增长
    p3  技术权衡：枚举 O(2^m) vs 逐项阈值的运行时间
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np

from sim.population import make_instance, make_groups
from sim.mechanisms import MECHANISMS, LABELS_ZH, run_mechanism
from sim.metrics import evaluate, METRIC_LABELS_ZH
from sim import paper_demo as P
from sim import plots

HERE = os.path.dirname(os.path.abspath(__file__))
MECHS = list(MECHANISMS)


# ----------------------------------------------------------------------------- 工具函数
def check_feasible(X, inst):
    assert (X.sum(axis=1) <= inst.k).all(), "有学生超过 k 门"
    assert (X.sum(axis=0) <= inst.cap).all(), "有课程超过容量"


def run_one(cfg, seed, groups=None):
    """同一个场景下依次运行五种机制。每种机制有独立的随机数流，保证可复现。"""
    rng = np.random.default_rng(seed)
    inst = make_instance(cfg, rng, groups=groups)
    rows, norms, extra = {}, {}, {}
    for idx, name in enumerate(MECHS):
        X, info = run_mechanism(name, inst, np.random.default_rng([seed, idx, 7]), cfg)
        check_feasible(X, inst)
        rows[name], norms[name] = evaluate(inst, X, cfg["k"])
        if name == "FCFS":
            extra["fill_time"] = info["fill_time"]
            extra["popularity"] = inst.popularity
    return rows, norms, extra


def aggregate(list_of_rows):
    """[{mech: {metric: v}}] -> {mech: {metric: (mean, ci95)}}"""
    out = {}
    for mech in MECHS:
        out[mech] = {}
        for metric in list_of_rows[0][mech]:
            vals = np.array([r[mech][metric] for r in list_of_rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            mean = float(vals.mean()) if vals.size else float("nan")
            ci = float(1.96 * vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            out[mech][metric] = (mean, ci)
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig：Excel/WPS 打开不乱码
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(v, d=2, pct=False):
    """数值标签：避免出现 “-0.00”。"""
    v = 0.0 if round(v * (100 if pct else 1), d) == 0 else v
    return f"{v:.{d}%}" if pct else f"{v:.{d}f}"


def seeds(cfg, n):
    return list(range(n))


# ----------------------------------------------------------------------------- 实验
def exp_e1(cfg, n_seeds, out, fig, summary):
    rows, fills = [], []
    for s in seeds(cfg, n_seeds):
        r, _, extra = run_one(cfg, s)
        rows.append(r)
        order = np.argsort(-extra["popularity"])
        fills.append(extra["fill_time"][order])           # 按热门程度排序后的“抢光时间”
    agg = aggregate(rows)
    table = []
    for mech in MECHS:
        for metric, (m, ci) in agg[mech].items():
            table.append([mech, LABELS_ZH[mech], metric, METRIC_LABELS_ZH.get(metric, metric), round(m, 4), round(ci, 4)])
    write_csv(os.path.join(out, "e1_main.csv"), ["mechanism", "机制", "metric", "指标", "mean", "ci95"], table)

    F = np.array(fills)
    top5 = F[:, :5]
    summary["e1"] = {m: {k: round(v[0], 4) for k, v in agg[m].items()} for m in MECHS}
    summary["e1_ci95"] = {m: {k: round(v[1], 4) for k, v in agg[m].items()} for m in MECHS}
    summary["e1_fcfs_fill"] = {
        "top5_popular_median_fill_s": round(float(np.nanmedian(top5)), 2),
        "all_courses_median_fill_s": round(float(np.nanmedian(F)), 2),
        "share_courses_full_within_60s": round(float(np.mean(np.nan_to_num(F, nan=np.inf) <= 60)), 3),
    }

    plots.setup()
    import matplotlib.pyplot as plt
    panels = ["satisfaction_mean", "satisfaction_p10", "zero_rate", "envy_mean_norm", "gap_busy", "gap_slow_net"]
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    names = [LABELS_ZH[m] for m in MECHS]
    for ax, metric in zip(axes.ravel(), panels):
        means = [agg[m][metric][0] for m in MECHS]
        cis = [agg[m][metric][1] for m in MECHS]
        x = np.arange(len(MECHS))
        ax.bar(x, means, width=0.55, color=plots.SERIES, yerr=cis,
               error_kw={"elinewidth": 0.8, "ecolor": plots.INK2, "capsize": 2})
        for xi, v, c in zip(x, means, cis):
            y = v + c if v >= 0 else v - c
            ax.annotate(fmt(v), (xi, y), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=8, color=plots.INK)
        ax.axhline(0, color=plots.AXIS, linewidth=0.8)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_title(METRIC_LABELS_ZH[metric])
        plots.style_axis(ax)
    figure.suptitle(f"五种选课机制对比（{cfg['n_students']}名学生、{cfg['n_courses']}门课、每人{cfg['k']}门、座位/需求={cfg['seat_ratio']}，{n_seeds}次随机重复，误差线为95%置信区间）",
                    fontsize=10, color=plots.INK2)
    figure.tight_layout()
    figure.savefig(os.path.join(fig, "fig_e1_main.png"))
    plt.close(figure)


def sweep(cfg, key, values, n_seeds, metrics):
    res = {m: {met: [] for met in metrics} for m in MECHS}
    ci = {m: {met: [] for met in metrics} for m in MECHS}
    for v in values:
        c = copy.deepcopy(cfg)
        c[key] = v
        agg = aggregate([run_one(c, s)[0] for s in seeds(c, n_seeds)])
        for m in MECHS:
            for met in metrics:
                res[m][met].append(agg[m][met][0])
                ci[m][met].append(agg[m][met][1])
    return res, ci


def line_panels(values, res, metrics, xlabel, title, path, logx=False, xticks=None):
    plots.setup()
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(metrics), figsize=(3.9 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    for ax, met in zip(axes, metrics):
        for color, m in zip(plots.SERIES, MECHS):
            ax.plot(values, res[m][met], color=color, label=LABELS_ZH[m], marker="o", markersize=4,
                    markeredgecolor=plots.SURFACE, markeredgewidth=1.0)
        ax.set_title(METRIC_LABELS_ZH[met])
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log", base=2)
        if xticks is not None:
            ax.set_xticks(xticks, [str(t) for t in xticks])
        plots.style_axis(ax, ygrid_only=False)
    axes[0].legend(loc="best")
    figure.suptitle(title, fontsize=10, color=plots.INK2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def exp_e2(cfg, n_seeds, out, fig, summary):
    values = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    metrics = ["gap_busy", "zero_rate", "satisfaction_p10"]
    res, ci = sweep(cfg, "seat_ratio", values, n_seeds, metrics)
    rows = [[m, LABELS_ZH[m], met, v, round(res[m][met][i], 4), round(ci[m][met][i], 4)]
            for m in MECHS for met in metrics for i, v in enumerate(values)]
    write_csv(os.path.join(out, "e2_scarcity.csv"), ["mechanism", "机制", "metric", "seat_ratio", "mean", "ci95"], rows)
    summary["e2"] = {"seat_ratio": values, **{m: {met: [round(x, 4) for x in res[m][met]] for met in metrics} for m in MECHS}}
    line_panels(values, res, metrics, "座位数 / 选课需求", f"座位越紧张，机制差异越大（每点 {n_seeds} 次重复；先到先得与纯抽签在中、右两图中几乎重合）",
                os.path.join(fig, "fig_e2_scarcity.png"), xticks=values)


def exp_e3(cfg, n_seeds, out, fig, summary):
    values = [1, 2, 4, 8, 16]
    metrics = ["satisfaction_mean", "top1_rate"]
    c = copy.deepcopy(cfg)
    c["sigma"] = 0.4
    res, ci = sweep(c, "tau", values, n_seeds, metrics)
    c0 = copy.deepcopy(cfg)
    c0["sigma"] = 0.0
    agg0 = aggregate([run_one(c0, s)[0] for s in seeds(c0, n_seeds)])
    rows = [[m, LABELS_ZH[m], met, v, round(res[m][met][i], 4), round(ci[m][met][i], 4)]
            for m in MECHS for met in metrics for i, v in enumerate(values)]
    rows += [[m, LABELS_ZH[m], met, "无噪声", round(agg0[m][met][0], 4), round(agg0[m][met][1], 4)]
             for m in MECHS for met in metrics]
    write_csv(os.path.join(out, "e3_noise_queries.csv"), ["mechanism", "机制", "metric", "tau", "mean", "ci95"], rows)
    summary["e3"] = {"sigma": 0.4, "tau": values,
                     **{m: {met: [round(x, 4) for x in res[m][met]] for met in metrics} for m in MECHS},
                     "noise_free": {m: {met: round(agg0[m][met][0], 4) for met in metrics} for m in MECHS}}
    line_panels(values, res, metrics, "每门课的偏好“查询”次数 tau（σ=0.4）",
                f"学生越了解自己的偏好，分配越好（每点 {n_seeds} 次重复）",
                os.path.join(fig, "fig_e3_noise_queries.png"), logx=True, xticks=values)


def exp_e4(cfg, n_seeds, out, fig, summary, semesters=8, bad=0.5):
    """连续 8 个学期。“网络慢/用脚本”固定（设备与网络通常不会每学期变化）；
    “开放时正忙”分两种情形：每学期重新抽样（临时冲突） vs 固定不变（如长期兼职、固定课表冲突）。"""
    conds = {"temporary": "“正忙”每学期随机（临时冲突）", "persistent": "“正忙”固定不变（如长期兼职）"}
    res = {}
    for cond in conds:
        share_bad, persist = {m: [] for m in MECHS}, {m: [] for m in MECHS}
        for s in seeds(cfg, n_seeds):
            rng = np.random.default_rng([s, 99])
            base = make_groups(cfg["n_students"], cfg["p_slow_net"], cfg["p_busy"], cfg["p_script"], rng)
            norms = {m: [] for m in MECHS}
            for t in range(semesters):
                g = dict(base)
                if cond == "temporary":
                    g["busy"] = rng.random(cfg["n_students"]) < cfg["p_busy"]
                _, nm, _ = run_one(cfg, 10_000 * (s + 1) + t, groups=g)
                for m in MECHS:
                    norms[m].append(nm[m])
            for m in MECHS:
                A = np.array(norms[m])                        # (semesters, n)
                bad_count = (A < bad).sum(axis=0)
                share_bad[m].append(float((bad_count >= semesters // 2).mean()))
                C = np.corrcoef(A)                            # 不同学期之间、同一批学生结果的相关
                persist[m].append(float(C[np.triu_indices(semesters, 1)].mean()))
        res[cond] = {"share_students_bad_half_or_more": {m: round(float(np.mean(share_bad[m])), 4) for m in MECHS},
                     "between_semester_corr": {m: round(float(np.mean(persist[m])), 4) for m in MECHS}}
    summary["e4"] = {"semesters": semesters, "bad_threshold": bad, **res}
    rows = [[cond, m, LABELS_ZH[m], res[cond]["share_students_bad_half_or_more"][m], res[cond]["between_semester_corr"][m]]
            for cond in conds for m in MECHS]
    write_csv(os.path.join(out, "e4_semesters.csv"),
              ["condition", "mechanism", "机制", f"≥{semesters // 2}个学期满意度<{bad}的学生比例", "学期间结果相关系数"], rows)

    plots.setup()
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    x = np.arange(len(MECHS))
    names = [LABELS_ZH[m] for m in MECHS]
    for ax, cond in zip(axes, conds):
        vals = [res[cond]["share_students_bad_half_or_more"][m] for m in MECHS]
        corr = [res[cond]["between_semester_corr"][m] for m in MECHS]
        ax.bar(x, vals, width=0.55, color=plots.SERIES)
        for xi, v, c in zip(x, vals, corr):
            ax.annotate(f"{fmt(v, 1, pct=True)}\nr={fmt(c)}", (xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=plots.INK)
        ax.set_xticks(x, names, rotation=20, ha="right")
        ax.set_title(conds[cond])
        plots.style_axis(ax)
    axes[0].set_ylabel(f"≥{semesters // 2} 个学期满意度 < {bad} 的学生比例")
    top = max(max(res[c]["share_students_bad_half_or_more"].values()) for c in conds)
    axes[0].set_ylim(0, top * 1.35)
    figure.suptitle(f"连续 {semesters} 个学期：吃亏的是不是总是同一批人？（r = 同一学生不同学期结果的平均相关系数，{n_seeds} 次重复）",
                    fontsize=9.5, color=plots.INK2)
    figure.tight_layout()
    figure.savefig(os.path.join(fig, "fig_e4_semesters.png"))
    plt.close(figure)


def exp_p1(n_trials, out, fig, summary):
    rng = np.random.default_rng(2026)
    taus = [1, 2, 4, 8, 16, 32, 64]
    small, gap = P.success_vs_queries(12, taus, n_trials, 1.0, rng, include_brute=True)
    large, _ = P.success_vs_queries(100, taus, n_trials, 1.0, rng)
    summary["p1"] = {"tau": taus, "sigma": 1.0, "m12_mean_gap_delta": round(gap, 3),
                     "m12": {k: [round(v, 3) for v in r] for k, r in small.items()},
                     "m100": {k: [round(v, 3) for v in r] for k, r in large.items()}}
    rows = [[m_, k, t, v] for m_, R in (("12", small), ("100", large)) for k, r in R.items() for t, v in zip(taus, r)]
    write_csv(os.path.join(out, "p1_success.csv"), ["m", "method", "tau", "success_rate"], rows)

    plots.setup()
    import matplotlib.pyplot as plt
    lab = {"threshold_tuned": "逐项阈值（调 c，论文算法思路）", "brute_force": "枚举全部分配（论文算法 1）",
           "threshold_c1": "逐项阈值（固定 c = 1）", "random": "完全随机分配（不查询）"}
    col = {"threshold_tuned": plots.SERIES[0], "brute_force": plots.SERIES[1],
           "threshold_c1": plots.SERIES[2], "random": plots.SERIES[3]}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, R, title in ((axes[0], small, f"m = 12 件物品（平均最优负嫉妒 Δ≈{gap:.2f}）"), (axes[1], large, "m = 100 件物品")):
        for k in ["threshold_tuned", "brute_force", "threshold_c1", "random"]:
            if k in R:
                ax.plot(taus, R[k], color=col[k], label=lab[k], marker="o", markersize=4,
                        markeredgecolor=plots.SURFACE, markeredgewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(taus, [str(t) for t in taus])
        ax.set_xlabel("每件物品的查询次数 τ（噪声 σ = 1）")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        plots.style_axis(ax, ygrid_only=False)
    axes[0].set_ylabel("得到真正无嫉妒分配的比例")
    axes[0].legend(loc="upper left")
    figure.suptitle(f"两人噪声查询模型：查询越多越能找到无嫉妒分配，但必须“平衡”两人（每点 {n_trials} 次试验；a 对物品的估值普遍高于 b）",
                    fontsize=9.5, color=plots.INK2)
    figure.tight_layout()
    figure.savefig(os.path.join(fig, "fig_p1_success.png"))
    plt.close(figure)


def exp_p2(n_trials, out, fig, summary):
    rng = np.random.default_rng(7)
    ms = [25, 50, 100, 200, 400, 800]
    r = P.random_allocation_fluctuation(ms, n_trials, rng)
    summary["p2"] = {k: [round(float(x), 4) for x in v] for k, v in r.items()}
    write_csv(os.path.join(out, "p2_random_fluctuation.csv"), ["m", "envy_mean", "envy_std", "p_envy_free"],
              list(zip(r["m"], r["envy_mean"], r["envy_std"], r["p_ef"])))
    # m = 100 时的嫉妒分布
    hist = []
    for _ in range(n_trials):
        ua, ub = rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)
        hist.append(P.envies(P.random_allocation(100, rng), ua, ub)[0])

    plots.setup()
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].hist(hist, bins=40, color=plots.SERIES[0], edgecolor=plots.SURFACE, linewidth=0.6)
    axes[0].axvline(0, color=plots.INK2, linewidth=1)
    axes[0].set_title("m = 100：完全随机分配下 a 对 b 的嫉妒分布")
    axes[0].set_xlabel("Envy(a→b)（> 0 表示 a 嫉妒 b）")
    axes[0].set_ylabel("次数")
    plots.style_axis(axes[0])
    m_arr = np.array(r["m"], dtype=float)
    axes[1].plot(m_arr, r["envy_std"], color=plots.SERIES[0], marker="o", markersize=5,
                 markeredgecolor=plots.SURFACE, markeredgewidth=1.0, label="实测：嫉妒的标准差")
    ref = r["envy_std"][0] * np.sqrt(m_arr / m_arr[0])
    axes[1].plot(m_arr, ref, color=plots.MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="参考：∝ √m")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    from matplotlib.ticker import NullFormatter, NullLocator
    for axis in (axes[1].xaxis, axes[1].yaxis):
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks(ms, [str(x) for x in ms])
    yt = [3, 4, 6, 8, 12, 16]
    axes[1].set_yticks(yt, [str(y) for y in yt])
    axes[1].set_ylabel("Envy(a→b) 的标准差")
    axes[1].set_xlabel("物品数 m")
    axes[1].set_title(f"波动随 √m 增长；两人都不嫉妒的概率仅约 {np.mean(r['p_ef']):.0%}")
    axes[1].legend(loc="upper left")
    plots.style_axis(axes[1], ygrid_only=False)
    figure.suptitle(f"“抽签”在平均意义上公平（均值≈0），但单次结果波动很大（{n_trials} 次试验）", fontsize=10, color=plots.INK2)
    figure.tight_layout()
    figure.savefig(os.path.join(fig, "fig_p2_random.png"))
    plt.close(figure)


def exp_p3(out, fig, summary):
    rng = np.random.default_rng(11)
    ms = [6, 8, 10, 12, 14, 16, 18, 20]
    r = P.runtime_brute_vs_threshold(ms, rng)
    slope = np.polyfit(r["m"][-4:], np.log2(r["brute_s"][-4:]), 1)
    est40 = 2 ** np.polyval(slope, 40)
    summary["p3"] = {**{k: [round(float(x), 6) for x in v] for k, v in r.items()},
                     "brute_extrapolated_m40_hours": round(float(est40 / 3600), 1)}
    write_csv(os.path.join(out, "p3_runtime.csv"), ["m", "brute_force_s", "threshold_s"],
              list(zip(r["m"], r["brute_s"], r["threshold_s"])))
    plots.setup()
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(r["m"], r["brute_s"], color=plots.SERIES[1], marker="o", markersize=5, markeredgecolor=plots.SURFACE, label="枚举全部 2^m 种分配")
    ax.plot(r["m"], r["threshold_s"], color=plots.SERIES[0], marker="o", markersize=5, markeredgecolor=plots.SURFACE, label="逐项阈值（含 c 的网格搜索）")
    ax.set_yscale("log")
    ax.set_xlabel("物品数 m")
    ax.set_ylabel("运行时间（秒，对数坐标）")
    ax.set_title(f"技术权衡：枚举法每多 1 件物品时间约翻倍（按本机速度外推 m=40 约需 {est40 / 3600:.0f} 小时）")
    ax.legend(loc="upper left")
    plots.style_axis(ax, ygrid_only=False)
    figure.tight_layout()
    figure.savefig(os.path.join(fig, "fig_p3_runtime.png"))
    plt.close(figure)


# ----------------------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description="选课分配机制模拟器")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "default.json"))
    ap.add_argument("--exp", nargs="*", default=["all"])
    ap.add_argument("--quick", action="store_true", help="少量重复，用于快速检查")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--fig", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    cfg = {k: v for k, v in json.load(open(a.config, encoding="utf-8")).items() if not k.startswith("_")}
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.fig, exist_ok=True)
    exps = {"e1", "e2", "e3", "e4", "p1", "p2", "p3"} if "all" in a.exp else set(a.exp)
    ns = 4 if a.quick else cfg["seeds"]
    ns_sweep = 3 if a.quick else 20
    trials = 40 if a.quick else 300
    summary_path = os.path.join(a.out, "summary.json")
    summary = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else {}
    summary["config"] = cfg
    summary["quick"] = a.quick
    t0 = time.time()
    steps = [("e1", lambda: exp_e1(cfg, ns, a.out, a.fig, summary)),
             ("e2", lambda: exp_e2(cfg, ns_sweep, a.out, a.fig, summary)),
             ("e3", lambda: exp_e3(cfg, ns_sweep, a.out, a.fig, summary)),
             ("e4", lambda: exp_e4(cfg, ns_sweep, a.out, a.fig, summary)),
             ("p1", lambda: exp_p1(trials, a.out, a.fig, summary)),
             ("p2", lambda: exp_p2(trials * 4, a.out, a.fig, summary)),
             ("p3", lambda: exp_p3(a.out, a.fig, summary))]
    for name, fn in steps:
        if name in exps:
            t = time.time()
            fn()
            print(f"[{name}] 完成，用时 {time.time() - t:.1f} 秒")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"全部完成，用时 {time.time() - t0:.1f} 秒。结果在 {a.out}，图在 {a.fig}")


if __name__ == "__main__":
    main()

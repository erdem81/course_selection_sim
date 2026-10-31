"""评价指标面板。全部用真实效用 U 计算（机制做决定时只看到带噪声的 B）。

Metrics panel, always computed with the TRUE utilities U.
第1章确定“用哪几个指标、达到多少算解决”之后，第2章只报告对应的那几项即可。
"""
import numpy as np


def student_values(U, X, k):
    """每个学生实际得到的总效用、理想总效用（不受容量限制时的前 k 门）、二者之比。"""
    value = (U * X).sum(axis=1)
    best = -np.sort(-U, axis=1)[:, :k].sum(axis=1)
    norm = np.where(best > 0, value / best, 0.0)
    return value, best, norm


def envy_stats(U, X, value, best):
    """i 嫉妒 j：i 认为 j 的课表（按 i 的真实效用）比自己的好。与论文的 Envy 定义一致（加性效用）。"""
    V = U @ X.T.astype(float)              # V[i, j] = i 对 j 课表的估值
    np.fill_diagonal(V, -np.inf)
    max_other = V.max(axis=1)
    envy = np.maximum(0.0, max_other - value)
    envy_norm = np.where(best > 0, envy / best, 0.0)
    envious = envy > 1e-9
    return envious.mean(), float(envy_norm.mean()), float((envy_norm > 0.25).mean())


def jain(x):
    x = np.asarray(x, dtype=float)
    s2 = (x ** 2).sum()
    return float(x.sum() ** 2 / (len(x) * s2)) if s2 > 0 else 1.0


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(inst, X, k):
    """返回一个指标字典。group_gap_* = 该群体平均满意度 − 其他人平均满意度（负数 = 吃亏）。"""
    U = inst.U
    value, best, norm = student_values(U, X, k)
    fav = np.argmax(U, axis=1)
    top1 = X[np.arange(len(fav)), fav]
    envy_rate, mean_envy, severe = envy_stats(U, X, value, best)
    out = {
        "satisfaction_mean": float(norm.mean()),          # 平均满意度（实际/理想）
        "satisfaction_p10": float(np.percentile(norm, 10)),  # 最差 10% 学生的满意度
        "top1_rate": float(top1.mean()),                   # 选上真实最爱课程的比例
        "full_k_rate": float((X.sum(axis=1) >= k).mean()), # 选满 k 门的比例
        "zero_rate": float((X.sum(axis=1) == 0).mean()),   # 一门都没选上的比例
        "envy_rate": float(envy_rate),                     # 至少嫉妒一个人的学生比例
        "envy_mean_norm": mean_envy,                       # 平均（最大）嫉妒量 / 理想效用
        "envy_severe_rate": severe,                        # 嫉妒量超过理想效用 25% 的学生比例
        "jain": jain(norm),
        "gini": gini(norm),
        "speed_outcome_corr": spearman(-inst.arrival, norm),  # 越早到 → 越满意？（Spearman）
    }
    for g, mask in inst.groups.items():
        if mask.any() and (~mask).any():
            out[f"gap_{g}"] = float(norm[mask].mean() - norm[~mask].mean())
        else:
            out[f"gap_{g}"] = float("nan")
    return out, norm


METRIC_LABELS_ZH = {
    "satisfaction_mean": "平均满意度",
    "satisfaction_p10": "最差10%学生满意度",
    "top1_rate": "选上最想要课程的比例",
    "full_k_rate": "选满课程的比例",
    "zero_rate": "一门都没选上的比例",
    "envy_rate": "存在嫉妒的学生比例",
    "envy_mean_norm": "平均嫉妒量(归一化)",
    "envy_severe_rate": "严重嫉妒学生比例(>25%)",
    "jain": "Jain公平指数",
    "gini": "基尼系数",
    "speed_outcome_corr": "到达速度与满意度的相关",
    "gap_slow_net": "网络慢群体差距",
    "gap_busy": "开放时正忙群体差距",
    "gap_script": "脚本用户差距",
}

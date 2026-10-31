"""五种选课分配机制。所有机制只使用学生的自我认知/上报偏好 B，不使用真实效用 U。

Five allocation mechanisms. Every mechanism sees only B (beliefs / reports), never U.

返回值 X: (n, m) bool 矩阵，X[i, j] = 学生 i 选上了课程 j。
约束：每行之和 ≤ k；每列之和 ≤ cap[j]；同一门课一个学生最多一个座位。
假设：学生如实按 B 行动（不做策略性虚报）——这是模型的边界之一。
"""
import numpy as np


def _best_available(b_row, remaining, held_row):
    """在“还有座位且自己没选过”的课里，返回 B 最高的一门；没有则返回 -1。"""
    ok = (remaining > 0) & ~held_row
    if not ok.any():
        return -1
    scores = np.where(ok, b_row, -np.inf)
    return int(np.argmax(scores))


def serial_dictatorship(B, cap, k, order, times=None):
    """按给定顺序，每个人一次性拿走自己最想要的 k 门（还有座位的）。

    times: 可选，(n,) 每人的到达时间；给出时同时返回每门课“被抢光”的时间。
    """
    n, m = B.shape
    X = np.zeros((n, m), dtype=bool)
    remaining = cap.astype(int).copy()
    fill_time = np.full(m, np.nan)
    for i in order:
        for _ in range(k):
            j = _best_available(B[i], remaining, X[i])
            if j < 0:
                break
            X[i, j] = True
            remaining[j] -= 1
            if remaining[j] == 0 and times is not None:
                fill_time[j] = times[i]
    return (X, fill_time) if times is not None else X


def fcfs(inst, rng=None):
    """先到先得：按到达时间排序的序列独裁（serial dictatorship）。"""
    order = np.argsort(inst.arrival, kind="stable")
    X, fill_time = serial_dictatorship(inst.B, inst.cap, inst.k, order, times=inst.arrival)
    return X, {"fill_time": fill_time, "order": order}


def rsd_lottery(inst, rng):
    """纯抽签（随机序列独裁 RSD）：随机打乱顺序，再按顺序一次性选 k 门。"""
    order = rng.permutation(inst.B.shape[0])
    return serial_dictatorship(inst.B, inst.cap, inst.k, order), {"order": order}


def snake_draft(inst, rng):
    """蛇形轮选（draft）：随机顺序；第 1 轮每人选 1 门，第 2 轮倒序再选 1 门，……共 k 轮。"""
    B, cap, k = inst.B, inst.cap, inst.k
    n, m = B.shape
    X = np.zeros((n, m), dtype=bool)
    remaining = cap.astype(int).copy()
    order = rng.permutation(n)
    for r in range(k):
        seq = order if r % 2 == 0 else order[::-1]
        for i in seq:
            j = _best_available(B[i], remaining, X[i])
            if j >= 0:
                X[i, j] = True
                remaining[j] -= 1
    return X, {"order": order}


def _fill_leftovers(X, B, remaining, k, rng):
    """补选：随机顺序，让还没选满 k 门的人从剩余座位中按偏好补齐。"""
    n = B.shape[0]
    for i in rng.permutation(n):
        while X[i].sum() < k:
            j = _best_available(B[i], remaining, X[i])
            if j < 0:
                break
            X[i, j] = True
            remaining[j] -= 1
    return X


def points_bidding(inst, rng, top_L=4, points=100):
    """意愿点投标：每人有 points 点，按偏好强度成比例投给前 top_L 门；
    全体投标从高到低处理，有座位且未选满就录取；最后随机顺序补选。

    投标策略假设为“按偏好比例如实投”，现实中学生会策略性投标（见 Sönmez & Ünver, 2010）。
    """
    B, cap, k = inst.B, inst.cap, inst.k
    n, m = B.shape
    top = np.argsort(-B, axis=1)[:, :top_L]
    w = np.clip(np.take_along_axis(B, top, axis=1), 1e-6, None)
    bids = points * w / w.sum(axis=1, keepdims=True)
    students = np.repeat(np.arange(n), top_L)
    courses = top.ravel()
    amounts = bids.ravel() + rng.uniform(0, 1e-9, size=bids.size)  # 随机打破平局
    X = np.zeros((n, m), dtype=bool)
    remaining = cap.astype(int).copy()
    for idx in np.argsort(-amounts, kind="stable"):
        i, j = students[idx], courses[idx]
        if remaining[j] > 0 and X[i].sum() < k and not X[i, j]:
            X[i, j] = True
            remaining[j] -= 1
    X = _fill_leftovers(X, B, remaining, k, rng)
    return X, {}


def ranked_lottery(inst, rng, list_L=4):
    """志愿 + 抽签：每人按偏好填 list_L 个志愿；第 r 轮只看第 r 志愿，
    报名人数不超过剩余座位则全录，否则随机抽取；L 轮后随机顺序补选。
    （即“即时接受 / 波士顿机制 + 抽签”。）
    """
    B, cap, k = inst.B, inst.cap, inst.k
    n, m = B.shape
    ranks = np.argsort(-B, axis=1)[:, :list_L]
    X = np.zeros((n, m), dtype=bool)
    remaining = cap.astype(int).copy()
    for r in range(list_L):
        need = X.sum(axis=1) < k
        choice = ranks[:, r]
        for j in np.unique(choice[need]):
            applicants = np.where(need & (choice == j) & ~X[:, j])[0]
            if remaining[j] <= 0 or applicants.size == 0:
                continue
            if applicants.size > remaining[j]:
                applicants = rng.choice(applicants, size=remaining[j], replace=False)
            X[applicants, j] = True
            remaining[j] -= applicants.size
    X = _fill_leftovers(X, B, remaining, k, rng)
    return X, {}


MECHANISMS = {
    "FCFS": fcfs,
    "Lottery(RSD)": rsd_lottery,
    "SnakeDraft": snake_draft,
    "PointsBidding": points_bidding,
    "RankedLottery": ranked_lottery,
}

# 图表与报告中使用的中文名
LABELS_ZH = {
    "FCFS": "先到先得",
    "Lottery(RSD)": "纯抽签(RSD)",
    "SnakeDraft": "蛇形轮选",
    "PointsBidding": "意愿点投标",
    "RankedLottery": "志愿+抽签",
}


def run_mechanism(name, inst, rng, cfg):
    fn = MECHANISMS[name]
    if name == "PointsBidding":
        return fn(inst, rng, top_L=cfg["bid_top_L"], points=cfg["bid_points"])
    if name == "RankedLottery":
        return fn(inst, rng, list_L=cfg["rank_list_L"])
    return fn(inst, rng)

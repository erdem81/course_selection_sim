"""生成模拟选课场景：学生、课程、真实效用、带噪声的自我认知、FCFS 到达时间。

Generate a synthetic course-selection instance.

真实效用 U[i, j] ∈ [0, 1]：学生 i 对课程 j 的真实满意度（模拟器知道，学生自己并不确切知道）。
自我认知 B[i, j]：学生 i 以为的满意度 = U + 噪声（论文中的“加性高斯噪声查询”，取 tau 次平均）。
所有机制都只能看到 B（学生的选择/上报），而评价指标一律用 U 计算。
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Instance:
    U: np.ndarray              # (n, m) 真实效用
    B: np.ndarray              # (n, m) 学生的自我认知 / 上报偏好
    cap: np.ndarray            # (m,) 每门课容量
    k: int                     # 每人最多选几门
    arrival: np.ndarray        # (n,) FCFS 中开放后到达的秒数
    groups: dict = field(default_factory=dict)  # 名称 -> (n,) bool，例如 slow_net / busy / script
    popularity: np.ndarray = None               # (m,) 课程热门程度


def make_capacities(n, m, k, seat_ratio, rng):
    """把总座位数 round(seat_ratio * n * k) 尽量平均地分到 m 门课。"""
    total = int(round(seat_ratio * n * k))
    base = total // m
    cap = np.full(m, base, dtype=int)
    cap[: total - base * m] += 1
    rng.shuffle(cap)
    return cap


def make_utilities(n, m, w_pop, rng):
    """U = w_pop * 课程热门度 + (1 - w_pop) * 个人兴趣，裁剪到 [0, 1]。"""
    popularity = rng.beta(2.0, 2.0, size=m)
    taste = rng.uniform(0.0, 1.0, size=(n, m))
    U = w_pop * popularity[None, :] + (1.0 - w_pop) * taste
    return np.clip(U, 0.0, 1.0), popularity


def noisy_beliefs(U, sigma, tau, rng):
    """tau 次独立噪声观测 y = u + N(0, sigma^2) 的平均值，方差为 sigma^2 / tau。"""
    if sigma <= 0:
        return U.copy()
    return U + rng.normal(0.0, sigma / np.sqrt(max(tau, 1)), size=U.shape)


def make_groups(n, p_slow_net, p_busy, p_script, rng):
    """与偏好无关的属性：网络慢、开放时正忙、用脚本。三者独立抽样（脚本用户不再算慢网）。"""
    script = rng.random(n) < p_script
    slow_net = (rng.random(n) < p_slow_net) & ~script
    busy = rng.random(n) < p_busy
    return {"slow_net": slow_net, "busy": busy, "script": script}


def make_arrival(groups, cfg, rng):
    """FCFS 到达时间（秒）：反应时间(对数正态) + 慢网额外延迟 + 正忙延迟；脚本用户几乎瞬时。"""
    n = len(groups["slow_net"])
    t = rng.lognormal(mean=np.log(cfg["reaction_median_s"]), sigma=0.6, size=n)
    t = t + groups["slow_net"] * rng.exponential(cfg["slow_net_extra_s"], size=n)
    busy_delay = rng.uniform(cfg["busy_delay_min_s"], cfg["busy_delay_max_s"], size=n)
    t = t + groups["busy"] * busy_delay
    t = np.where(groups["script"], rng.uniform(0.0, cfg["script_reaction_s"], size=n), t)
    return t


def make_instance(cfg, rng, groups=None):
    """按配置生成一个完整场景。groups 可传入固定的学生属性（多学期实验用）。"""
    n, m, k = cfg["n_students"], cfg["n_courses"], cfg["k"]
    cap = make_capacities(n, m, k, cfg["seat_ratio"], rng)
    U, popularity = make_utilities(n, m, cfg["w_pop"], rng)
    B = noisy_beliefs(U, cfg["sigma"], cfg["tau"], rng)
    if groups is None:
        groups = make_groups(n, cfg["p_slow_net"], cfg["p_busy"], cfg["p_script"], rng)
    arrival = make_arrival(groups, cfg, rng)
    return Instance(U=U, B=B, cap=cap, k=k, arrival=arrival, groups=groups, popularity=popularity)

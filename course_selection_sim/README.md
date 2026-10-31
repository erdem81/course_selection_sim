# 选课分配机制模拟器（工程导论大作业 C16）

> **AI 使用标注**：本仓库代码初稿由 Claude（Anthropic）生成，2026-10-21。
> 负责人：李信。每一处人工修改都要记入《AI使用日志》（提示词 / AI 给了什么 / 改了哪里 / 为什么改）。

## 这个模拟器做什么

在同一批“虚拟学生 + 课程”上，比较五种选课分配方式，用**学生的真实满意度**打分：

| 代号 | 机制 | 一句话说明 |
|---|---|---|
| `FCFS` | 先到先得 | 按到达时间排队，先到的人一次拿走最想要的 k 门（现行“抢课”） |
| `Lottery(RSD)` | 纯抽签 | 随机排队，其余同上（随机序列独裁） |
| `SnakeDraft` | 蛇形轮选 | 随机排队，每轮每人只选 1 门，下一轮倒序 |
| `PointsBidding` | 意愿点投标 | 每人 100 点按偏好强度投给几门课，出价高者先得 |
| `RankedLottery` | 志愿 + 抽签 | 第 r 轮只看第 r 志愿，报名超额则抽签 |

建模要点（与文献的联系）：

- 学生对每门课有**真实效用** `U`，但学生自己也只知道一个带噪声的估计
  `B = U + N(0, σ²/τ)`——这就是论文 *Envy-Free Allocation of Indivisible Goods via Noisy Queries*（ICML 2026）
  的加性高斯噪声查询模型，τ 是“查询”次数。
- 所有机制只能看到 `B`；所有指标都用 `U` 计算。
- 先到先得中，到达时间受**与偏好无关**的因素影响：网络/设备慢、开放时正在上课或打工、使用抢课脚本。
- 嫉妒（envy）的定义与论文相同：学生 i 用自己的真实效用评价 j 的课表，若高于自己的课表，则 i 嫉妒 j。

## 运行方法

需要 Python 3.9+。

```bash
pip install -r requirements.txt
python tests/test_mechanisms.py          # 6 项正确性检查
python run_experiments.py                # 全部实验，约 1 分钟，生成 results/ 和 figures/
python run_experiments.py --quick        # 快速检查（约 10 秒）
python run_experiments.py --exp e1 p1    # 只跑部分实验
python run_experiments.py --config config/my.json   # 用问卷校准后的参数
```

所有随机过程使用固定种子，同一配置重复运行得到相同结果（`p3` 的运行时间除外，它取决于电脑速度）。

## 参数与校准

参数在 `config/default.json`，每个参数都有中文说明。**标注 `[待问卷校准]` 的数值是假设值**：
第 1 章的问卷 / 访谈完成后，用真实数据（例如“开放时正在上课的同学比例”“热门课几秒被抢光”）替换，
另存为 `config/my.json` 再运行。实验 e1 会输出先到先得下热门课的“抢光时间”，可直接和问卷数据对照。

## 实验与输出

| 实验 | 问题 | 输出 |
|---|---|---|
| e1 | 默认参数下五种机制的全部指标 | `results/e1_main.csv`，`figures/fig_e1_main.png` |
| e2 | 座位越紧张，差异如何变化 | `results/e2_scarcity.csv`，`figures/fig_e2_scarcity.png` |
| e3 | 学生越了解自己偏好（τ 越大），结果如何变化 | `results/e3_noise_queries.csv`，`figures/fig_e3_noise_queries.png` |
| e4 | 连续 8 学期，吃亏的是不是总是同一批人 | `results/e4_semesters.csv`，`figures/fig_e4_semesters.png` |
| p1 | 论文两人模型：阈值规则 / 枚举 / 随机分配的成功率 | `results/p1_success.csv`，`figures/fig_p1_success.png` |
| p2 | 随机分配“平均公平”但波动 ∝ √m | `results/p2_random_fluctuation.csv`，`figures/fig_p2_random.png` |
| p3 | 技术权衡：枚举 O(2^m) vs 阈值规则的运行时间 | `results/p3_runtime.csv`，`figures/fig_p3_runtime.png` |

所有关键数字汇总在 `results/summary.json`。CSV 用 UTF-8-BOM 编码，可直接用 Excel / WPS 打开。

指标（`sim/metrics.py`）：平均满意度、最差 10% 学生满意度、选上最想要课程的比例、选满比例、一门都没选上的比例、
存在嫉妒的比例、平均嫉妒量、严重嫉妒比例、Jain 公平指数、基尼系数、到达速度与满意度的相关、各群体差距。
**第 1 章确定“用哪几个指标、达到多少算解决”后，第 2 章只需报告对应几项。**

## 目录结构

```
config/default.json     参数（含中文说明）
sim/population.py       生成学生、课程、真实效用、带噪声偏好、FCFS 到达时间
sim/mechanisms.py       五种分配机制
sim/metrics.py          指标面板
sim/paper_demo.py       论文两人模型的演示实验
sim/plots.py            统一作图样式
run_experiments.py      实验入口
tests/test_mechanisms.py 正确性检查
```

## 模型的边界（不做什么）

- 假设学生**如实**按自己的估计行动，不模拟策略性填报 / 投标（意愿点投标和志愿抽签在现实中都会被策略性利用）。
- 不模拟服务器拥塞、选课系统崩溃、退补选阶段、课程时间冲突与先修要求。
- 效用是加性的：两门课的满意度 = 各自满意度之和。
- 论文演示实验（p1–p3）只用于直观说明，不是对论文理论界的数值验证。

## 参考

- Zihan Li, Yan Hao Ling, Jonathan Scarlett, Warut Suksompong. Envy-Free Allocation of Indivisible Goods via Noisy Queries. *Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*, PMLR 306, 2026. arXiv:2602.06361.
- Eric Budish, Estelle Cantillon. The Multi-unit Assignment Problem: Theory and Evidence from Course Allocation at Harvard. *American Economic Review*, 102(5): 2237–2271, 2012.
- Tayfun Sönmez, M. Utku Ünver. Course Bidding at Business Schools. *International Economic Review*, 51(1): 99–123, 2010.

# 微网与外部电网电力调控策略优化模型（2026 国赛 C 题）

以 10 分钟为调度粒度（全天 144 段），用**线性规划 + 滚动时域优化(MPC)**求解含光伏与储能的非孤岛微网购电与充放电策略，覆盖题目全部四问。

## 运行环境

- Python 3.14（3.10+ 亦可）
- 依赖：`pip install -r requirements.txt`（numpy / scipy(含 HiGHS) / pandas / openpyxl / matplotlib / python-docx / pypdf）

## 数据准备

将题目附件 `附件1.xlsx～附件4.xlsx` 与结果模板 `result1.xlsx、result2.xlsx、result3.xlsx、result4-2.xlsx、result4-3.xlsx`
放到同一数据目录，并把 `common.py` 顶部的 `DESK` 变量改为该目录的绝对路径。

## 代码结构与运行顺序

| 文件 | 作用 |
|---|---|
| `common.py` | 物理常量、数据加载、附件3 整点预报到 10 分钟的线性插值 |
| `io_utils.py` / `writer.py` | 结果表读写、按模板循环移位写出工具 |
| `optimizer.py` | 通用线性规划求解器（功率平衡、SOC 递推、容量/功率约束），HiGHS 求解 |
| `mpc.py` | 滚动时域窗口优化与单日 MPC（计划—调整—紧急购电、调整费用线性化） |
| `solve_q1.py` | 问题一：单天确定性计划购电 |
| `solve_q2.py` | 问题二：固定电价全年连续优化 |
| `solve_q3.py` / `exp_blend.py` | 问题三：滚动调整模型与 latest/min/avg 预报融合对比 |
| `generate_q3.py` / `generate_q4.py` | 生成 result3、result4-2、result4-3 |
| `produce_all.py` | 汇总 `results_summary.json` 与论文用图 |
| `verify_all.py` | 独立回读校验（行和、电费重算、SOC 越界、负购电等） |
| `paper_lib.py` / `build_paper.py` | 按国赛格式生成论文 Word（三线表、页码、附录源码） |
| `probe_files.py / probe2.py / probe3.py / probe4.py` | 附件结构探查脚本 |

推荐顺序：`solve_q1 → solve_q2 → solve_q3(exp_blend) → generate_q3 → generate_q4 → produce_all → verify_all → build_paper`。

## 主要结果（2025-02-01～12-31）

- 问题一：全天购电 59482.70 kWh、购电费 35118.60 元，首末 SOC 均为 8550 kWh；
- 问题二（固定电价）：购电费 1222.95 万元，确定性下紧急购电为 0；
- 问题三：保守滚动调整将紧急购电量由 60.10 万 kWh 降至 14.72 万 kWh（-75.5%）；
- 问题四（波动电价）：问题二购电费 1278.26 万元，问题三结论方向一致。

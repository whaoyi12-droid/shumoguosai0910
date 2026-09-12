# 第三问 v4 单文件运行与复现

## 范围

将现有 v4 联合版和两个消融配置需要的函数整理到 `src/solve_question3_v4_standalone.py`，保留原预测、MILP、执行、退款结算和一月预热的运算顺序。原始代码和既有结果不覆盖。单文件包含独立物理/结算审计、断点续算和逐日结果对照，无项目内模块依赖。

原始附件仍是输入；程序不包含现成结果。暂不包含 Excel 提交模板排版，现有 v4 数值成果是 CSV/JSON。

## 负荷预测与滚动窗口核对

v4 沿用问题二的同星期加权历史均值与趋势修正算法，但不沿用问题二的三日展望。当前预测和优化每 6 小时更新一次，展望未来 24 小时（144 段），执行前 6 小时（36 段）。这一窗口在 v3 六小时版已经采用，v4 保持不变。

`load_forecast` 分别调用 `predict_load_day(d, 0, ...)` 和 `predict_load_day(d, 1, ...)`，生成今天与明天的日曲线，再用 `[slot:slot + 144]` 截取未来 24 小时。拼接两天是为跨日窗口取数，不是同时优化两天或三天。例如当天 06:00 的展望为当天 06:00 至次日 06:00。

基础预测只使用当天之前最多 31 天的历史，优先取同星期样本，采用 0.92 的时间衰减和 0.3 的趋势修正。日内 06/12/18 点新增信息通过已观测负荷前缀与预测前缀的比值修正当天剩余部分，比值限制为 0.8–1.2；明天部分保持基线预测。关闭 `prefix-ratio` 时仍是 24 小时展望，只是不做日内负荷校正。

原共享模块 `question3_forecast.py` 中保留的 `LOOKAHEAD_DAYS=3` 与 `forecast_three_days` 属于历史接口，v4 的当前预测/优化调用链不调用该三日接口。单文件版没有该三日函数或常量。“最近 3 条残差场景”中的 3 是场景条数，每条长度仍为 24 小时。

核对位置：单文件的 `predict_load_day`、`load_forecast`、`Predictor.scenarios` 和 `simulate_day`；模块版为 `question3_forecast_v4.py` 的 `load_forecast` 与 `solve_question3_v4.py` 的 `simulate_day`。本次为静态调用链审查，没有改动或重跑数值模型。

## 运行

将该 `.py` 与 `附件1.xlsx`、`附件2.xlsx`、`附件3.xlsx` 放在同一目录，运行：

```powershell
python -X utf8 solve_question3_v4_standalone.py
```

也可将附件放进同目录的 `附件` 子目录，或使用 `--base-dir "数据目录"`。默认在数据目录的 `results/q3_v4_standalone` 输出，不覆盖已有目录。

原成果环境为 Python 3.12.4、numpy 2.1.3、pandas 3.0.2、scipy 1.17.1、openpyxl 3.1.5；换环境尤其换 SciPy/HiGHS 可能产生最优解选择或浮点差异。

在需要准备的 Python 3.12.4 环境中，依赖版本为：

```powershell
python -m pip install numpy==2.1.3 pandas==3.0.2 scipy==1.17.1 openpyxl==3.1.5
```

本次使用已有环境，未安装或改动全局依赖。传输时不需要原来的其他 `.py`，也不需要 `.artifact_work` 内的开发用提取脚本。

```powershell
# 联合版；reference 只在计算完成后读取，用于核验，不参与求解。
python -X utf8 solve_question3_v4_standalone.py --base-dir "数据目录" --output "新输出目录" --reference "原q3_v4_combined目录"
# 仅负荷校正
python -X utf8 solve_question3_v4_standalone.py --pv-interpolation linear --output "新的负荷消融目录"
# 仅插值选择
python -X utf8 solve_question3_v4_standalone.py --load-correction none --output "新的插值消融目录"
# 中断后继续；配置必须与原运行一致
python -X utf8 solve_question3_v4_standalone.py --output "已有输出目录" --resume
# 不求解，只审计此单文件生成的结果
python -X utf8 solve_question3_v4_standalone.py --output "已有输出目录" --audit-only
```

## 验证口径

新输出目录重算，逐日比较预热与正式期的全部账本字段，以及四次更新的预测、场景、决策和求解证书。运行耗时和源码哈希不要求相等。报告同时记录绝对差、是否数值完全相等、逐日 CSV 是否字节相等；不能用总费用接近代替逐时一致。

已完成的短程验证：

- 单文件导入后重跑原 v4 的 7 项合成测试，全部通过：未来观测/预报扰动隔离、成熟窗口、负荷校正边界、插值节点、关闭功能兼容性和执行器无主动浪费。
- 使用 `python -I` 隔离项目搜索路径，从原始附件重算一月预热与二月前两天。26 份逐日 CSV 字节一致，预测、场景、决策和非耗时求解证书数值最大差为 0；104 次求解独立审计通过。
- 将脚本与三个原始附件复制到 `results/q3_v4_standalone_portability/`，不指定 `--base-dir`、不携带其他代码运行成功。续算和独立审计通过，26 份逐日 CSV 仍字节一致。续算重读 CSV 后的汇总计算可能存在约 1e-12 的浮点舍入差，这是原版续算机制的既有行为。
- 静态检查无未定义全局引用；原八个依赖源码和三个附件哈希未变。

独立目录续算已扩展到二月三日，108 次求解审计通过，27 份逐日 CSV 与原版字节一致；新增一天的预测、场景与决策也逐项相等。

## 全年复现已完成

2026-09-12 在上述固定环境中，从原始附件重新运行三个配置。每组包含一月 24 天预热和二月至十二月 334 天正式期，正式期 48,096 条记录、全程 1,432 次 MILP。三组独立审计均通过。

| 配置 | 新输出目录（results 下） | 总费用/元 | 应急电量/kWh | 与原版数值最大差 |
|---|---|---:|---:|---:|
| 联合版（默认） | q3_v4_standalone_combined | 13223253.262282858 | 29357.488867956636 | 0 |
| 仅负荷校正 | q3_v4_standalone_load | 13223666.162733445 | 29429.47732686592 | 0 |
| 仅插值选择 | q3_v4_standalone_interpolation | 13228816.309536576 | 38222.92609208296 | 0 |

每组 358 份逐日账本 CSV 均与对应原版字节相同。全年账本、预测、场景、调度决策、费用、SOC、校准结果、汇总及非耗时求解证书逐项数值相同。源码哈希与求解耗时变化属于预期，未将其作为数值复现失败。全部比较使用真实新运行输出，原版结果只在求解完成后被运行入口读取并比较。

证据为各新目录的 `validation.json`、`reproducibility.json`，统一核验记录为 `results/q3_v4_standalone_verification.json`。7 项回归、隔离目录运行/续算及原源码和附件哈希不变均纳入统一记录。

交付源码为 UTF-8 文本，共 55,817 字节；SHA256 为 `cbde241f76512b269591c36d125e2492430f64c2083a01721c4a1923793020b2`。运行和校验均只需该单文件及原始附件，不依赖本次复现报告。

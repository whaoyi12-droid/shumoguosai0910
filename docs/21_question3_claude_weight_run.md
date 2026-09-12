# 交给Claude：第三问月度权重全年对照

本交接只要求执行现有代码、运行独立校验、整理实际结果，不要求Claude修改核心模型、权重规则或求解精度。当前模型为每6小时更新、向前24小时、取消量净退回半价，主目标为总费用最小。冷启动及月度权重依据见20号文档，调度基础见19号文档。

## 1. 这次已经准备好的内容

- 1月1日0、6、12点的预测器只使用最新预报；18点后使用有效版本融合。最新预报不当作真值。原1月前7天空闲、1月8日起预热策略不变。
- `solve_question3_six_hour.py`现为v3，支持`--weight-mode`：

| 参数 | 含义 |
|---|---|
| `january` | 原1月4–30日固定权重，默认，保留原基线 |
| `january-full` | 统一口径的完整1月成熟样本权重，全年固定 |
| `previous-month` | 每月初用上个月成熟样本拟合 |
| `expanding` | 每月初用截至上月底累计成熟样本拟合 |

同一月的参数冻结。训练目标必须在月初前成熟；例如2月28日18点发布、到3月1日18点才完成的24小时目标，不能用于3月1日0点的对应权重。程序不提供当月事后权重模式。

每个输出目录有`weight_schedule.json`；每次发布版本另存实际使用的权重、有效月份及训练末时间。历史残差仍按当前冻结参数对过去数据重构，为参数内回测误差，不是另存的原历史预测；需要保留这一不确定性估计的近似说明。

## 2. 执行范围与命令

优先完成两种全年方案，用相同的1月样本和状态预热隔离“每月更新”的影响。不要拿`january`与`expanding`的所有差异都解释成更新频率差异，因为旧1月校准样本数不同。

```powershell
Set-Location -LiteralPath 'C:\Users\y7000\Desktop\Mine\华中杯\2026.9.10-9.13国赛\C题'

python -X utf8 src/solve_question3_six_hour.py --weight-mode january-full --output results/q3_monthly_fixed
python -X utf8 src/solve_question3_six_hour.py --weight-mode expanding --output results/q3_monthly_expanding

python -X utf8 src/validate_question3_six_hour_run.py results/q3_monthly_fixed
python -X utf8 src/validate_question3_six_hour_run.py results/q3_monthly_expanding

python -X utf8 src/compare_question3_weight_modes.py results/q3_monthly_fixed results/q3_monthly_expanding --output results/q3_monthly_weight_review/annual_dispatch_comparison.csv
```

每条命令成功后再执行下一步。结果目录须尚不存在。若同一版本中断，只在原命令末尾加`--resume`；不要直接对v2或其他权重模式的旧目录恢复。源码、附件、参数的哈希不符时换新目录并记录原因，不修改检查点绕过校验。

如需要验证“只用上个月”的策略，可另运行第三组，不替换前两组：

```powershell
python -X utf8 src/solve_question3_six_hour.py --weight-mode previous-month --output results/q3_monthly_previous
python -X utf8 src/validate_question3_six_hour_run.py results/q3_monthly_previous
python -X utf8 src/compare_question3_weight_modes.py results/q3_monthly_fixed results/q3_monthly_expanding results/q3_monthly_previous --output results/q3_monthly_weight_review/annual_dispatch_comparison_three_modes.csv
```

## 3. 结果导出

两组都完成334天且审计通过后，可分别导出模板副本；原附件5不覆盖：

```powershell
python -X utf8 src/prepare_question3_six_hour_workbook.py results/q3_monthly_fixed
& 'C:\Users\y7000\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' src/write_question3_six_hour_workbook.mjs results/q3_monthly_fixed

python -X utf8 src/prepare_question3_six_hour_workbook.py results/q3_monthly_expanding
& 'C:\Users\y7000\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' src/write_question3_six_hour_workbook.mjs results/q3_monthly_expanding
```

各目录单独生成`result3.xlsx`。现有Node/artifact-tool依赖沿用项目环境；不可自行安装全局依赖。若导出环境不可用，保留已验证账本和payload并报告具体错误，不能伪称已完成工作簿。导出后仍要把各表的电量、退款后费用、SOC及应急区间与账本核对。旧版工作簿通过的证据不代表本版工作簿也已通过。

## 4. 验收和停止条件

- 2月1日至12月31日共334天、48096个实际十分钟段；每6小时一个版本，时域144段，执行36段。
- 保留1月8–31日预热文件。两种权重方案的2月初SOC须一致；年末SOC允许不同，但必须一起报告，不能忽略存量电能差异。
- 每次MILP的相对gap≤1e-4，或符合代码明确的绝对gap≤1e-6退化兜底；不得提高容差或接受未认证解来完成全年。失败保存`last_failure.json`、出错日期和原始日志后停止，交回用户。
- 独立审计须核对原始净负载、电价、能量平衡、SOC和模式、退款结算、原计划及实际采用版本、月度权重的成熟时间、文件哈希。
- 不将光伏RMSE改善直接换算成节省电费，不强行要求动态权重获胜。报告总费、计划费、增购费、退款额、应急费/量、余量、首末SOC及相对固定基线变化。
- 不改第二/第四问，不覆盖旧results，不自行调权重、终值参数或预测器，不写未经验证的论文结论。

返回用户：两个`summary.json`、两个`validation.json`、全年对照CSV、工作簿路径与核对证据；若失败，返回失败文件和已完成日期范围。主agent后续再核验关键产物。

## 5. 已完成的准备验证

- 7项预测/权重测试通过，包括前三次冷启动、第四次加权、未来数据扰动、目标成熟时间、月度切换和3月四次求解的权重来源。
- 已分别运行`results/q3_weight_fixed_smoke`与`results/q3_weight_expanding_smoke`，各为1月预热+2月1–2日，共104次MILP；独立审计通过，最大gap7.537e-5、最大账本残差约5.46e-12。
- 两组2月初/2日末SOC均10800kWh，两日总费均78435.50515元、退款2470.79998元、应急0。这一相同结果验证2月可用信息相同，不是对动态全年优势的证明。
- 预报全年回测与调度全年运行是不同任务：前者已完成，后者留给本次Claude执行。完整报告见20号文档。本版尚未生成新的全年工作簿。

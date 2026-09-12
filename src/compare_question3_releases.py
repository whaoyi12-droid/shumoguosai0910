"""Audit every Q3 release-hour variant and tabulate the value of intraday forecast updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from validate_question3_annual import audit, ROOT

VARIANTS = [('q3_zero_only', '仅 0:00'), ('q3_rel_0_6', '0:00, 6:00'), ('q3_rel_0_12', '0:00, 12:00'),
            ('q3_rel_0_18', '0:00, 18:00'), ('q3_three_day', '0:00, 6:00, 12:00, 18:00')]


def main(folder, base):
    rows, audits = [], {}
    for name, label in VARIANTS:
        path = folder / name
        report, _ = audit(path, base)
        audits[name] = report
        (path / 'annual_audit.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        rows.append({'方案': label, '发布次数/日': len(label.split(',')) ,'总费用': report['annual_cost'],
                     '计划费': report['cost_breakdown']['plan_cost'],
                     '增购费': report['cost_breakdown']['increase_cost'],
                     '减购费': report['cost_breakdown']['reduction_cost'],
                     '应急费': report['cost_breakdown']['emergency_cost'],
                     '应急量_kWh': report['annual_emergency_kwh'],
                     '弃余量_kWh': report['annual_surplus_kwh'],
                     '首SOC': report['initial_soc'], '末SOC': report['final_soc'],
                     '问题数': len(report['issues']), '校验残差': max(report['max_residuals'].values())})
    table = pd.DataFrame(rows)
    table['相对仅0点节约_元'] = table['总费用'].iloc[0] - table['总费用']
    table['相对仅0点节约_%'] = table['相对仅0点节约_元'] / table['总费用'].iloc[0] * 100
    return table, audits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--base-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    table, _ = main(args.results, args.base_dir)
    pd.set_option('display.width', 260)
    print(table.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    table.to_csv(args.results / 'q3_release_comparison.csv', index=False)

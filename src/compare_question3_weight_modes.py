"""Audit then compare weight policies under matched dates and starting stored energy."""
import argparse
from pathlib import Path

import pandas as pd

from validate_question3_six_hour_run import audit


def compare(folders,allow_partial=False):
    reports=[audit(folder) for folder in folders]
    reference=reports[0]
    for report in reports:
        if not report['annual_complete'] and not allow_partial:
            raise ValueError('Full-year comparison requires 334 days; --allow-partial is diagnostic only')
        if ((report['start_date'],report['end_date'])!=(reference['start_date'],reference['end_date'])
                or abs(report['initial_soc']-reference['initial_soc'])>1e-5):
            raise ValueError('Unmatched dates or initial SOC')
        if any(report[k]!=reference[k] for k in ('newest_only','adjustments','settlement')):
            raise ValueError('Forecast/transaction rules differ; this is not an isolated weight-policy comparison')
    rows=[]
    for report,folder in zip(reports,folders):
        rows.append({'folder':str(folder),'weight_mode':report['weight_mode'],'days':report['days'],
            **{k:report[k] for k in ('total_cost','refund_amount','emergency_kwh','surplus_kwh','initial_soc','final_soc')},
            'saving_vs_first':reference['total_cost']-report['total_cost'],
            'emergency_reduction_vs_first':reference['emergency_kwh']-report['emergency_kwh']})
    return pd.DataFrame(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders',nargs='+',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    table=compare(args.folders,args.allow_partial)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output,index=False)
    print(table.round(5).to_string(index=False))

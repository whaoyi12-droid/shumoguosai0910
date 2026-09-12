"""Monthly PV-weight drift and causal monthly refits; no dispatch or price optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_question3_revision import calibrate, fit_monotone, vintage_matrix
from solve_question3 import ROOT, digest, load_inputs, save_json


def collect_samples(inputs):
    _,_,pv,dates,releases=inputs
    actual=pv.ravel()
    blocks=[]
    for block,n in enumerate((4,3,2,1)):
        blocks.append({'x':[], 'y':[], 'issue':[], 'available_at':[]})
    for issue in sorted(releases):
        matrix=vintage_matrix(issue,pv,dates,releases)
        flat=int((issue-dates[0]).total_seconds()/600)
        for block,n in enumerate((4,3,2,1)):
            a,b=block*36,(block+1)*36
            if flat+b>len(actual) or not np.isfinite(matrix[a:b,:n]).all():
                continue
            record=blocks[block]
            record['x'].append(matrix[a:b,:n])
            record['y'].append(actual[flat+a:flat+b])
            record['issue'].append(issue)
            record['available_at'].append(issue+pd.Timedelta(hours=6*(block+1)))
    for record in blocks:
        record['x']=np.asarray(record['x'])
        record['y']=np.asarray(record['y'])
        record['issue']=pd.DatetimeIndex(record['issue'])
        record['available_at']=pd.DatetimeIndex(record['available_at'])
    return blocks


def select(record,start,cutoff):
    # Cutoff is an absolute decision time. Every target in an included block
    # must already have been observed; a prior release alone is insufficient.
    return ((record['issue']>=pd.Timestamp(start)) & (record['issue']<pd.Timestamp(cutoff))
            & (record['available_at']<=pd.Timestamp(cutoff)))


def fit_record(record,mask):
    x=record['x'][mask]
    if not len(x):
        raise ValueError('No mature training samples')
    return fit_monotone(x.reshape(-1,x.shape[-1]),record['y'][mask].ravel())


def fit_asof(blocks,start,cutoff):
    result=[]
    for record in blocks:
        mask=select(record,start,cutoff)
        weights=fit_record(record,mask)
        result.append({'weights':weights.tolist(),'training_blocks':int(mask.sum()),
                       'training_points':int(mask.sum())*36,
                       'latest_available_at':str(record['available_at'][mask].max())})
    return result


def score(x,y,w):
    error=x@w-y
    return {'rmse_kw':float(np.sqrt(np.mean(error**2))),
            'mae_kw':float(np.mean(np.abs(error))),
            'sse':float(np.sum(error**2)),'sae':float(np.sum(np.abs(error))),'points':error.size}


def paired_bootstrap(daily,policy,reference='fixed_january_full',repeats=1000):
    """Paired, within-month circular seven-day blocks; positive delta means worse."""
    a=daily.loc[daily.policy==policy,['month','block','day','sse','points']]
    b=daily.loc[daily.policy==reference,['month','block','day','sse','points']]
    joined=a.merge(b,on=['month','block','day'],suffixes=('_new','_ref'),validate='one_to_one')
    rows=[];rng=np.random.default_rng(20250912)
    for block in range(4):
        groups=[g.sort_values('day') for _,g in joined.loc[joined.block==block].groupby('month')]
        boot=[]
        for _ in range(repeats):
            new=ref=count=0.
            for group in groups:
                n=len(group)
                starts=rng.integers(0,n,size=(n+6)//7)
                indices=np.concatenate([(np.arange(7)+s)%n for s in starts])[:n]
                sample=group.iloc[indices]
                new+=sample.sse_new.sum();ref+=sample.sse_ref.sum();count+=sample.points_new.sum()
            boot.append(float(np.sqrt(new/count)-np.sqrt(ref/count)))
        rows.append({'policy':policy,'reference':reference,'block':block,'replicates':repeats,
                     'rmse_delta_kw_ci95':np.quantile(boot,[.025,.975]).tolist()})
    return rows


def analyze(inputs,bootstrap_repeats=1000):
    blocks=collect_samples(inputs)
    legacy=[np.asarray(b['fit_weights']) for b in calibrate(inputs)['blocks']]
    january=fit_asof(blocks,'2025-01-01','2025-02-01')
    weight_rows=[];metric_rows=[];daily_rows=[];schedule={}
    for month in range(1,13):
        start=pd.Timestamp(2025,month,1);end=start+pd.offsets.MonthBegin(1)
        retro=fit_asof(blocks,start,end)
        if month>=2:
            previous=fit_asof(blocks,start-pd.offsets.MonthBegin(1),start)
            expanding=fit_asof(blocks,'2025-01-01',start)
            schedule[str(start.date())]={'decision_time':str(start),
                'previous_month':previous,'expanding_history':expanding,
                'not_current_month_oracle':True}
        for block,n in enumerate((4,3,2,1)):
            w=np.asarray(retro[block]['weights'])
            jan=np.asarray(january[block]['weights'])
            weight_rows.append({'month':month,'lead_start_hours':6*block,'lead_end_hours':6*(block+1),
                **{f'weight_age_{6*j}h':float(w[j]) if j<n else None for j in range(4)},
                'l1_change_vs_january_full':float(np.abs(w-jan).sum()),
                'max_abs_change_vs_january_full':float(np.abs(w-jan).max()),
                'training_points':retro[block]['training_points'],
                'fit_kind':'retrospective_month_oracle','latest_target':retro[block]['latest_available_at']})
            if month==1:
                continue
            record=blocks[block];mask=select(record,start,end)
            x,y=record['x'][mask],record['y'][mask]
            policies={'latest_only':np.r_[1.,np.zeros(n-1)],'fixed_january_legacy':legacy[block],
                      'fixed_january_full':jan,'previous_month':np.asarray(previous[block]['weights']),
                      'expanding_history':np.asarray(expanding[block]['weights']),
                      'current_month_oracle':w}
            for policy,weights in policies.items():
                metric=score(x,y,weights)
                metric_rows.append({'month':month,'block':block,'lead_hours':f'{6*block}-{6*(block+1)}',
                                    'policy':policy,**metric})
                error=x@weights-y
                temp=pd.DataFrame({'day':record['issue'][mask].strftime('%Y-%m-%d'),
                                   'sse':np.sum(error**2,axis=1),'points':36})
                for day,g in temp.groupby('day'):
                    daily_rows.append({'month':month,'block':block,'day':day,'policy':policy,
                                       'sse':float(g.sse.sum()),'points':int(g.points.sum())})
    metrics=pd.DataFrame(metric_rows)
    aggregate=[]
    for (policy,block),g in metrics.groupby(['policy','block'],sort=False):
        points=int(g.points.sum())
        aggregate.append({'policy':policy,'block':int(block),'points':points,
                          'rmse_kw':float(np.sqrt(g.sse.sum()/points)),'mae_kw':float(g.sae.sum()/points)})
    bootstrap=[]
    daily=pd.DataFrame(daily_rows)
    for policy in ('previous_month','expanding_history'):
        bootstrap.extend(paired_bootstrap(daily,policy,repeats=bootstrap_repeats))
    return pd.DataFrame(weight_rows),metrics,pd.DataFrame(aggregate),schedule,bootstrap


def main(args):
    inputs=load_inputs(args.base_dir)
    weights,metrics,aggregate,schedule,bootstrap=analyze(inputs,args.bootstrap_repeats)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    weights.to_csv(out/'monthly_weights.csv',index=False)
    metrics.to_csv(out/'monthly_forecast_metrics.csv',index=False)
    aggregate.to_csv(out/'aggregate_forecast_metrics.csv',index=False)
    save_json(out/'causal_monthly_weight_schedule.json',schedule)
    save_json(out/'bootstrap_intervals.json',bootstrap)
    save_json(out/'metadata.json',{'task':'forecast-only diagnostic, not dispatch results',
        'retrospective_weights_are_not_deployable_in_same_month':True,
        'evaluation':'Feb-Dec, same issue-month and target-maturity mask for all strategies',
        'bootstrap':'paired within-month circular 7-day resampling; seed 20250912',
        'bootstrap_repeats':args.bootstrap_repeats,
        'hashes':{p.name:digest(p) for p in [Path(__file__),args.base_dir/'附件/附件2.xlsx',args.base_dir/'附件/附件3.xlsx']}})
    print('NEXT SIX HOURS: monthly retrospective weights (newest first)')
    print(weights.loc[weights.lead_start_hours==0].round(5).to_string(index=False))
    print('FEB-DEC causal evaluations (oracle row is retrospective only)')
    print(aggregate.round(5).to_string(index=False))
    print(json.dumps(bootstrap,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'results/q3_monthly_weight_review')
    parser.add_argument('--bootstrap-repeats',type=int,default=1000)
    main(parser.parse_args())

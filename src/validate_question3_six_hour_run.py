"""Independent real-data physics, settlement, frozen-contract and monthly-weight audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_question3_annual import raw_inputs

ROOT=Path(__file__).resolve().parents[1]
TOL=1e-5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder,base=ROOT):
    price,load,pv,dates=raw_inputs(base)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if meta['model']!='Q3-six-hour-24h-MILP-refund-v3':
        raise ValueError('This audit requires the v3 weight-mode runner')
    for name,expected in meta['hashes'].items():
        path=base/('附件' if name.endswith('.xlsx') else 'src')/name
        if sha(path)!=expected:
            raise ValueError(f'Input/source hash changed since run: {name}')
    monthly=json.loads((folder/'weight_schedule.json').read_text(encoding='utf-8'))
    if monthly!=meta['weight_schedule']:
        raise ValueError('Weight schedule differs from run metadata')
    expected_dates=pd.date_range('2025-01-08',summary['end_date']).strftime('%Y-%m-%d').tolist()
    files=sorted(folder.glob('*_ledger.csv'))
    if [p.name[:10] for p in files]!=expected_dates:
        raise ValueError('Incomplete or extra dated ledger files')
    soc=6000.;frames=[];solves=0;max_gap=0.;worst={}
    def close(name,a,b):
        value=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        worst[name]=max(worst.get(name,0.),value)
        if not np.isfinite(value) or value>TOL:
            raise ValueError(f'{name} failed: {value}')
    for path in files:
        day=path.name[:10];d=int(dates.get_loc(day))
        cp=json.loads(path.with_name(f'{day}_complete.json').read_text(encoding='utf-8'))
        vf=path.with_name(f'{day}_versions.json')
        if sha(path)!=cp['ledger_sha256'] or sha(vf)!=cp['versions_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {day}')
        f=pd.read_csv(path)
        if f.date.tolist()!=[day]*144 or f.slot.tolist()!=list(range(1,145)):
            raise ValueError('Wrong day/slot keys')
        if not np.isfinite(f.select_dtypes('number')).all().all():
            raise ValueError('Nonfinite ledger')
        q,g,c,b,e,w=[f[n].to_numpy() for n in ('plan_kwh','final_regular_kwh','charge_kwh',
                                             'discharge_kwh','emergency_kwh','surplus_kwh')]
        start,end=f.soc_start_kwh.to_numpy(),f.soc_end_kwh.to_numpy()
        net=(load[d]-pv[d])/6
        close('raw_net',f.net_kwh,net);close('raw_price',f.price,price)
        close('balance',g+b+e,net+c+w);close('soc',end,start+.9*c-b/.9)
        close('soc_chain',start,np.r_[soc,end[:-1]]);soc=float(end[-1])
        if (min(q.min(),g.min(),c.min(),b.min(),e.min(),w.min()) < -TOL
                or min(start.min(),end.min()) < 1200-TOL or max(start.max(),end.max())>10800+TOL
                or max(c.max(),b.max())>5000/6+TOL or np.any((c>TOL)&((b>TOL)|(e>TOL)))):
            raise ValueError('Physical bounds/mode failed')
        amounts=np.c_[price*q,1.5*price*np.maximum(g-q,0),-.5*price*np.maximum(q-g,0),5*price*e]
        close('settlement',f[['plan_cost','increase_cost','reduction_cost','emergency_cost']],amounts)
        close('refund_amount',f.refund_amount,-amounts[:,2]);close('total',f.total_cost,amounts.sum(axis=1))
        versions=json.loads(vf.read_text(encoding='utf-8'))
        if [v['first_slot'] for v in versions]!=[1,37,73,109]:
            raise ValueError('Not four six-hour updates')
        close('frozen_q',q,versions[0]['regular_draft_kwh'])
        for v in versions:
            k=v['first_slot']-1
            if v['lookahead_slots']!=144 or v['executed_slots']!=36:
                raise ValueError('Wrong lookahead/execution length')
            if pd.Timestamp(v['issue_time'])!=pd.Timestamp(day)+pd.Timedelta(minutes=k*10):
                raise ValueError('Wrong issue timestamp')
            close('frozen_g',g[k:k+36],v['regular_draft_kwh'][:36])
            if d>=31:
                training=monthly[day[:7]]
                if v['weight_training']!=training:
                    raise ValueError('Wrong monthly training provenance')
                for actual,record in zip(v['weights'],training['blocks']):
                    close('weights',actual,record['weights'])
                    if pd.Timestamp(record['latest_available_at'])>pd.Timestamp(day[:7]+'-01'):
                        raise ValueError('Weight data not mature at month start')
            log=v['solver']
            if 'skipped' in log:
                if d<31 or meta['adjustments'] or k==0:
                    raise ValueError('Unexpected skipped solve')
                continue
            solves+=1
            relative=log['mip_gap'];absolute=log['absolute_gap']
            if (log['status']!=0 or not ((relative is not None and relative<=1e-4+1e-10) or absolute<=1e-6)
                    or log['max_constraint_violation']>TOL or log['simultaneous_charge_discharge'] or log['emergency_charge']):
                raise ValueError('Uncertified or physically invalid MILP')
            if relative is not None:
                max_gap=max(max_gap,relative)
        if d>=31:
            frames.append(f)
    annual=pd.concat(frames,ignore_index=True)
    saved=pd.read_csv(folder/'ledger.csv')
    if saved[['date','slot']].to_dict('list')!=annual[['date','slot']].to_dict('list'):
        raise ValueError('Combined ledger dates differ')
    close('combined_ledger',saved.select_dtypes('number'),annual.select_dtypes('number'))
    for name in ('plan_cost','increase_cost','reduction_cost','refund_amount','emergency_cost','total_cost',
                 'emergency_kwh','surplus_kwh'):
        close(f'summary_{name}',summary[name],annual[name].sum())
    close('summary_initial_soc',summary['initial_soc'],annual.soc_start_kwh.iloc[0])
    close('summary_final_soc',summary['final_soc'],soc)
    days=len(annual)//144
    if summary['days']!=days or summary['annual_complete']!=(days==334):
        raise ValueError('Incorrect completion status')
    report={'passed':True,'weight_mode':meta['weight_mode'],'days':days,'annual_complete':days==334,
            'newest_only':meta['newest_only'],'adjustments':meta['adjustments'],'settlement':meta['settlement'],
            'start_date':'2025-02-01','end_date':summary['end_date'],'initial_soc':summary['initial_soc'],
            'final_soc':soc,'solves':solves,'max_mip_gap':max_gap,'max_residuals':worst,
            **{n:float(annual[n].sum()) for n in ('plan_cost','increase_cost','refund_amount',
                'emergency_cost','total_cost','emergency_kwh','surplus_kwh')}}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('--base-dir',type=Path,default=ROOT)
    args=p.parse_args();report=audit(args.folder,args.base_dir)
    (args.folder/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

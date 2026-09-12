"""Audit prior Q3 output and calibrate causal, newest-first PV forecast blends."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from solve_question3 import ROOT, digest, load_inputs, save_json
from validate_question3_annual import audit


def vintage_matrix(issue, pv, dates, releases, horizon=144):
    """Rows=target ten-minute ends; columns=newest to oldest release, NaN if expired."""
    target = issue + pd.to_timedelta(np.arange(1, horizon + 1)*10, unit='min')
    values = np.full((horizon, 4), np.nan)
    for age in range(4):
        vintage = issue - pd.Timedelta(hours=6*age)
        if vintage not in releases:
            continue
        d = (vintage.normalize()-dates[0]).days
        slot = vintage.hour*6
        boundary = pv[d, slot-1] if slot else (pv[d-1, -1] if d else 0.)
        leads = np.asarray((target-vintage).total_seconds()/3600)
        valid = (leads > 0) & (leads <= 24)
        values[valid,age] = np.interp(leads[valid], np.arange(25), np.r_[boundary,releases[vintage]])
    return values


def fit_monotone(x,y):
    """Nonnegative simplex weights, constrained newest >= older."""
    n=x.shape[1]
    if n==1:
        return np.ones(1)
    scale=max(float(np.mean(y*y)),1.)
    gram=x.T@x/len(x)/scale
    xy=x.T@y/len(x)/scale
    result=minimize(lambda w: float(w@gram@w-2*xy@w), np.ones(n)/n,
                    jac=lambda w:2*(gram@w-xy), method='SLSQP',bounds=[(0,1)]*n,
                    constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)},
                                 {'type':'ineq','fun':lambda w:w[:-1]-w[1:]}],
                    options={'ftol':1e-12,'maxiter':500})
    if not result.success:
        raise RuntimeError(result.message)
    weights=np.maximum(result.x,0)
    return weights/weights.sum()


def calibrate(inputs):
    _,_,pv,dates,releases=inputs
    tables=[]
    for issue in sorted(releases):
        if not pd.Timestamp('2025-01-04') <= issue < pd.Timestamp('2025-01-31'):
            continue
        matrix=vintage_matrix(issue,pv,dates,releases)
        flat=(issue-dates[0]).total_seconds()/600
        actual=pv.ravel()[int(flat):int(flat)+144]
        for block,n in enumerate((4,3,2,1)):
            sl=slice(block*36,(block+1)*36)
            # Validation includes only targets inside January, independent of February.
            tables.append((issue,block,matrix[sl,:n],actual[sl]))
    report={'train':'2025-01-04--2025-01-20 releases; targets end by 2025-01-22',
            'validation':'2025-01-22--2025-01-30 releases; targets end by 2025-01-31 18:00',
            'final_fit':'2025-01-04--2025-01-30 releases, all targets observed before February',
            'blocks':[]}
    for block,n in enumerate((4,3,2,1)):
        train=[r for r in tables if r[1]==block and r[0]<pd.Timestamp('2025-01-21')]
        val=[r for r in tables if r[1]==block and r[0]>=pd.Timestamp('2025-01-22')]
        all_rows=[r for r in tables if r[1]==block]
        x,y=np.vstack([r[2] for r in train]),np.concatenate([r[3] for r in train])
        xv,yv=np.vstack([r[2] for r in val]),np.concatenate([r[3] for r in val])
        weights=fit_monotone(x,y)
        refit=fit_monotone(np.vstack([r[2] for r in all_rows]),np.concatenate([r[3] for r in all_rows]))
        candidates={'constrained_fit':weights,'newest_only':np.r_[1.,np.zeros(n-1)],
                    'equal':np.ones(n)/n,'q2_decay_0.92_per_6h':.92**np.arange(n)}
        scores={}
        for name,w in candidates.items():
            w=w/w.sum()
            error=xv@w-yv
            scores[name]={'weights':w.tolist(),'rmse_kw':float(np.sqrt(np.mean(error**2))),
                          'mae_kw':float(np.mean(np.abs(error)))}
        report['blocks'].append({'lead_hours':[block*6,(block+1)*6],'available_vintages':n,
             'fit_weights':refit.tolist(),'validation':scores})
    return report


def main():
    out=ROOT/'results/q3_revision_review'
    out.mkdir(exist_ok=True)
    inputs=load_inputs(ROOT)
    calibration=calibrate(inputs)
    save_json(out/'forecast_weights.json',calibration)
    reports={}
    for folder in ('q3_three_day','q3_zero_only'):
        report,frame=audit(ROOT/'results'/folder,ROOT)
        # Strengthen old audit: check version file hashes and frozen version-to-ledger matching.
        mismatches=[]
        for day in frame.date.unique():
            path=ROOT/'results'/folder/f'{day}_versions.json'
            checkpoint=json.loads(path.with_name(f'{day}_complete.json').read_text(encoding='utf-8'))
            if digest(path)!=checkpoint['versions_sha256']:
                mismatches.append(day)
            versions=json.loads(path.read_text(encoding='utf-8'))
            df=frame.loc[frame.date==day]
            np.testing.assert_allclose(df.plan_kwh,versions[0]['regular_draft_kwh'][:144],atol=1e-7,rtol=0)
            for j,v in enumerate(versions):
                start=v['first_slot']-1
                end=versions[j+1]['first_slot']-1 if j+1<len(versions) else 144
                np.testing.assert_allclose(df.final_regular_kwh.iloc[start:end],v['regular_draft_kwh'][:end-start],atol=1e-7,rtol=0)
        report['version_hash_mismatches']=mismatches
        report['actual_decrease_kwh']=float(np.maximum(frame.plan_kwh-frame.final_regular_kwh,0).sum())
        reports[folder]=report
    save_json(out/'prior_run_audit.json',reports)
    print(json.dumps({'calibration':calibration,'prior':{key:{k:r[k] for k in (
         'annual_cost','annual_emergency_kwh','annual_surplus_kwh','issues','lp_horizons','version_hash_mismatches')}
         for key,r in reports.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

"""Q3 v2: six-hour updates, 24-hour lookahead, causal PV blending, and half-price refunds."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import analyze_question3_revision as blend
import question3_forecast as forecast
import question3_weight_schedule as weight_schedule
import solve_question3 as previous

ROOT = previous.ROOT
LOW, HIGH, LIMIT, ETA, TOL = 1200., 10800., 5000/6, .9, 1e-5


def weight_sets(report=None):
    if report is None:
        return [(.92**np.arange(n))/(.92**np.arange(n)).sum() for n in (4,3,2,1)]
    return [np.asarray(block['fit_weights']) for block in report['blocks']]


def predict(d, slot, inputs, selected_k, weights, newest_only=False):
    _, load, pv, dates, releases = inputs
    issue=dates[d]+pd.Timedelta(minutes=10*slot)
    matrix=blend.vintage_matrix(issue,pv,dates,releases)
    # Jan 1 at 00/06/12 has not accumulated four releases. Latest forecast is
    # the causal startup estimate, never an assertion that future PV is known.
    cold_start=issue < dates[0]+pd.Timedelta(hours=18)
    solar=np.empty(144)
    for t in range(144):
        available=np.isfinite(matrix[t])
        w=np.zeros(4)
        if newest_only or cold_start:
            w[0]=1.
        else:
            w[:len(weights[t//36])]=weights[t//36]
        w[~available]=0.
        if w.sum()<=0:
            raise ValueError('No valid released forecast for target')
        solar[t]=float(np.nan_to_num(matrix[t])@(w/w.sum()))
    demand=np.concatenate([forecast.predict_load_day(d,off,load,dates,selected_k) for off in (0,1)])[slot:slot+144]
    return (demand-solar)/6


def make_scenarios(d, slot, inputs, selected_k, weights, cache=None, single=False, newest_only=False):
    base=predict(d,slot,inputs,selected_k,weights,newest_only)
    cache={} if cache is None else cache
    # A yesterday/same-hour 24h path has just matured, including today's observed prefix.
    origins=[] if single else list(range(max(3,d-45),d))[-3:]
    residuals=[]
    signature=tuple(tuple(float(x) for x in w) for w in weights)
    net=((inputs[1]-inputs[2])/6).ravel()
    for h in origins:
        key=(h,slot,selected_k,signature,newest_only)
        if key not in cache:
            start=h*144+slot
            historical=predict(h,slot,inputs,selected_k,weights,newest_only)
            cache[key]=net[start:start+144]-historical
        residuals.append(cache[key])
    paths=base[None,:]+np.asarray(residuals) if residuals else base[None,:]
    return paths,base,origins


def optimize(paths, prices, soc, lam, original=None, time_limit=60, gap=1e-4):
    """Shared regular purchases; strict battery physics for each scenario path."""
    paths=np.asarray(paths,dtype=float); prices=np.asarray(prices,dtype=float)
    m,horizon=paths.shape
    if (prices.shape!=(horizon,) or not np.isfinite(paths).all() or not np.isfinite(prices).all()
            or np.any(prices<=0) or not LOW<=soc<=HIGH):
        raise ValueError('Invalid MILP inputs')
    original=np.asarray([] if original is None else original,dtype=float)
    if len(original)>horizon or not np.isfinite(original).all() or np.any(original<0):
        raise ValueError('Invalid original plan')
    count=2*horizon+6*m*horizon
    cost=np.zeros(count); lo=np.zeros(count); hi=np.full(count,np.inf)
    integer=np.zeros(count,dtype=np.uint8)
    cost[:horizon]=prices
    hi[horizon:2*horizon]=0.
    def index(s,block,t):
        return 2*horizon+(6*s+block)*horizon+t
    rows=[];cols=[];vals=[]; lower=[];upper=[]
    def add(terms,low=-np.inf,high=np.inf):
        r=len(lower)
        for c,v in terms:
            rows.append(r);cols.append(c);vals.append(v)
        lower.append(low);upper.append(high)
    for t,q in enumerate(original):
        z=horizon+t
        hi[z]=np.inf;cost[z]=1.;cost[t]=0.
        add([(z,1),(t,-.5*prices[t])],low=.5*prices[t]*q)
        add([(z,1),(t,-1.5*prices[t])],low=-.5*prices[t]*q)
    for s in range(m):
        for t in range(horizon):
            c,b,e,w,state,y=[index(s,j,t) for j in range(6)]
            cost[c]=cost[b]=1e-5/m;cost[e]=5*prices[t]/m
            hi[c]=hi[b]=LIMIT;lo[state]=LOW;hi[state]=HIGH
            hi[y]=1;integer[y]=1
            nplus=max(paths[s,t],0.)
            hi[e]=nplus
            add([(t,1),(b,1),(e,1),(c,-1),(w,-1)],paths[s,t],paths[s,t])
            terms=[(state,1),(c,-ETA),(b,1/ETA)]
            if t:
                terms.append((index(s,4,t-1),-1))
            add(terms,soc if t==0 else 0,soc if t==0 else 0)
            add([(c,1),(y,-LIMIT)],high=0.)
            add([(b,1),(y,LIMIT)],high=LIMIT)
            add([(e,1),(y,nplus)],high=nplus)
        cost[index(s,4,horizon-1)]=-lam/m
    matrix=coo_matrix((vals,(rows,cols)),shape=(len(lower),count)).tocsc()
    began=time.perf_counter()
    result=milp(cost,integrality=integer,bounds=Bounds(lo,hi),
                constraints=LinearConstraint(matrix,lower,upper),
                options={'time_limit':time_limit,'mip_rel_gap':gap})
    if result.x is None or not np.isfinite(result.x).all() or result.status!=0:
        raise RuntimeError(f'MILP rejected: {result.status}: {result.message}')
    relative=float(result.mip_gap);absolute=abs(float(result.fun)-float(result.mip_dual_bound))
    if not ((np.isfinite(relative) and relative<=gap+1e-10) or absolute<=1e-6):
        raise RuntimeError('MILP optimality gap rejected')
    x=result.x; ax=matrix@x
    violation=max(float(np.max(np.maximum(np.asarray(lower)-ax,0))),
                  float(np.max(np.maximum(ax-np.asarray(upper),0))),
                  float(np.max(np.maximum(lo-x,0))),float(np.max(np.maximum(x-hi,0))),
                  float(np.max(np.abs(x[integer==1]-np.rint(x[integer==1])))))
    if violation>TOL:
        raise RuntimeError(f'Raw MILP vector failed constraints: {violation}')
    recourse=x[2*horizon:].reshape(m,6,horizon)
    simultaneous=int(np.sum((recourse[:,0]>TOL)&(recourse[:,1]>TOL)))
    emergency_charge=int(np.sum((recourse[:,0]>TOL)&(recourse[:,2]>TOL)))
    if simultaneous or emergency_charge:
        raise RuntimeError('MILP scenario physical mode violation')
    log={'status':int(result.status),'objective':float(result.fun),
         'mip_gap':relative if np.isfinite(relative) else None,'absolute_gap':absolute,
         'seconds':time.perf_counter()-began,'max_constraint_violation':violation,
         'horizon':horizon,'scenarios':m,'simultaneous_charge_discharge':simultaneous,
         'emergency_charge':emergency_charge}
    return np.maximum(x[:horizon],0),log


def simulate_day(d,soc,inputs,selected_k,weights,cache=None,single=False,newest_only=False,
                 adjustments=True,time_limit=60,weight_metadata=None):
    price,load,pv,dates,_=inputs
    lam=forecast.estimate_lambda(price)
    versions=[];rows=[];plan=None;current=None
    for slot in range(144):
        if slot%36==0:
            paths,base,origins=make_scenarios(d,slot,inputs,selected_k,weights,cache,single,newest_only)
            if slot==0 or adjustments:
                prices=np.tile(price,2)[slot:slot+144]
                solution,log=optimize(paths,prices,soc,lam,None if slot==0 else plan[slot:],time_limit)
                if slot==0:
                    plan=solution.copy();current=plan.copy()
                else:
                    current[slot:]=solution[:144-slot]
            else:
                # Identical information/update cadence; no permission to change regular purchases.
                solution=current[slot:].copy()
                log={'skipped':'no-adjustment baseline'}
            versions.append({'issue_time':str(dates[d]+pd.Timedelta(minutes=10*slot)),
                 'first_slot':slot+1,'executed_slots':36,'lookahead_slots':144,'soc':float(soc),
                 'selected_k':selected_k,'weights':[w.tolist() for w in weights],
                 'weight_training':weight_metadata,
                 'residual_dates':[str(dates[h].date()) for h in origins],
                 'forecast_kwh':base.tolist(),'scenarios_kwh':paths.tolist(),
                 'regular_draft_kwh':solution.tolist(),'lambda':lam,'solver':log})
        net=(load[d,slot]-pv[d,slot])/6
        c,b,e,w,end=previous.dispatch(net,current[slot],soc)
        pc=price[slot]*plan[slot]
        increase=1.5*price[slot]*max(current[slot]-plan[slot],0)
        refund=.5*price[slot]*max(plan[slot]-current[slot],0)
        emergency=5*price[slot]*e
        rows.append({'date':str(dates[d].date()),'slot':slot+1,
            'interval_start':str(dates[d]+pd.Timedelta(minutes=slot*10)),
            'interval_end':str(dates[d]+pd.Timedelta(minutes=(slot+1)*10)),
            'net_kwh':net,'plan_kwh':plan[slot],'final_regular_kwh':current[slot],
            'emergency_kwh':e,'charge_kwh':c,'discharge_kwh':b,'surplus_kwh':w,
            'soc_start_kwh':soc,'soc_end_kwh':end,'price':price[slot],
            'plan_cost':pc,'increase_cost':increase,'refund_amount':refund,'reduction_cost':-refund,
            'emergency_cost':emergency,'total_cost':pc+increase-refund+emergency})
        soc=end
    frame=pd.DataFrame(rows)
    verify_ledger(frame)
    return frame,versions,soc


def verify_ledger(df):
    if df.empty or len(df)%144 or not np.isfinite(df.select_dtypes('number')).all().all():
        raise ValueError('Incomplete or nonfinite ledger')
    for _,day in df.groupby('date',sort=False):
        if day.slot.tolist()!=list(range(1,145)):
            raise ValueError('Bad slot order')
    c,b,e,w=[df[n].to_numpy() for n in ('charge_kwh','discharge_kwh','emergency_kwh','surplus_kwh')]
    q,g,p=[df[n].to_numpy() for n in ('plan_kwh','final_regular_kwh','price')]
    start,end=df.soc_start_kwh.to_numpy(),df.soc_end_kwh.to_numpy()
    expected=np.c_[p*q,1.5*p*np.maximum(g-q,0),-.5*p*np.maximum(q-g,0),5*p*e]
    actual=df[['plan_cost','increase_cost','reduction_cost','emergency_cost']].to_numpy()
    error=max(float(np.max(np.abs(g+b+e-df.net_kwh-c-w))),
              float(np.max(np.abs(end-start-ETA*c+b/ETA))),float(np.max(np.abs(start[1:]-end[:-1]))),
              float(np.max(np.abs(expected-actual))),float(np.max(np.abs(expected.sum(axis=1)-df.total_cost))),
              float(np.max(np.abs(df.refund_amount+df.reduction_cost))))
    if (error>TOL or min(start.min(),end.min())<LOW-TOL or max(start.max(),end.max())>HIGH+TOL
            or min(c.min(),b.min(),e.min(),w.min(),q.min(),g.min()) < -TOL
            or max(c.max(),b.max())>LIMIT+TOL or np.any((c>TOL)&((b>TOL)|(e>TOL)))):
        raise ValueError('Physical or refund-accounting violation')
    return error


def run(args):
    inputs=previous.load_inputs(args.base_dir)
    _,load,pv,dates,_=inputs
    stop=pd.Timestamp(args.end_date)
    if stop not in dates or stop<pd.Timestamp('2025-02-01'):
        raise ValueError('Invalid end date')
    selected_k,k_table=forecast.calibrate_k_on_january(dates,load,pv)
    report=blend.calibrate(inputs)
    startup=weight_sets()
    schedule=weight_schedule.build_schedule(inputs,args.weight_mode,report)
    files=[Path(__file__),Path(blend.__file__),Path(previous.__file__),Path(forecast.__file__),
           Path(weight_schedule.__file__),ROOT/'src/analyze_question3_monthly_weights.py']+[
           args.base_dir/f'附件/附件{i}.xlsx' for i in (1,2,3)]
    meta={'model':'Q3-six-hour-24h-MILP-refund-v3','settlement':'50-percent-net-refund',
          'selected_k':selected_k,'weight_mode':args.weight_mode,'weight_schedule':schedule,
          'newest_only':args.newest_only,'adjustments':not args.no_adjustments,'time_limit':args.time_limit,
          'hashes':{p.name:previous.digest(p) for p in files}}
    out=args.output.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('Use a new output directory or matching --resume')
    out.mkdir(parents=True,exist_ok=True)
    mp=out/'metadata.json'
    if args.resume:
        if not mp.exists() or json.loads(mp.read_text(encoding='utf-8'))!=meta:
            raise ValueError('Resume metadata changed or missing')
    else:
        previous.save_json(mp,meta)
    previous.save_json(out/'forecast_weights.json',report)
    previous.save_json(out/'weight_schedule.json',schedule)
    k_table.to_csv(out/'k_calibration.csv',index=False)
    cache={};soc=6000.;frames=[]
    for d in range(7,int(dates.get_loc(stop))+1):
        day=str(dates[d].date())
        csv=out/f'{day}_ledger.csv';versions=out/f'{day}_versions.json';complete=out/f'{day}_complete.json'
        if complete.exists():
            checkpoint=json.loads(complete.read_text(encoding='utf-8'))
            if previous.digest(csv)!=checkpoint['ledger_sha256'] or previous.digest(versions)!=checkpoint['versions_sha256']:
                raise ValueError('Checkpoint file hash mismatch')
            frame=pd.read_csv(csv);verify_ledger(frame)
            if abs(soc-frame.soc_start_kwh.iloc[0])>TOL:
                raise ValueError('Resume SOC chain broken')
            soc=float(frame.soc_end_kwh.iloc[-1])
        else:
            try:
                training=({'mode':'startup-0.92','effective_at':'2025-01-01 00:00:00',
                           'latest_training_target':None} if d<31 else schedule[dates[d].strftime('%Y-%m')])
                current_weights=startup if d<31 else [np.asarray(b['weights']) for b in training['blocks']]
                frame,records,soc=simulate_day(d,soc,inputs,7 if d<31 else selected_k,
                     current_weights,cache,single=d<31,
                     newest_only=False if d<31 else args.newest_only,
                     adjustments=True if d<31 else not args.no_adjustments,time_limit=args.time_limit,
                     weight_metadata=training)
            except Exception as exc:
                previous.save_json(out/'last_failure.json',{'date':day,'error':str(exc)})
                raise
            frame.to_csv(csv,index=False)
            previous.save_json(versions,records)
            previous.save_json(complete,{'date':day,'ledger_sha256':previous.digest(csv),'versions_sha256':previous.digest(versions)})
            print(f'{day}: cost={frame.total_cost.sum():.2f}, refund={frame.refund_amount.sum():.2f}, emergency={frame.emergency_kwh.sum():.3f}',flush=True)
        if d>=31:
            frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    error=verify_ledger(result)
    result.to_csv(out/'ledger.csv',index=False)
    summary={n:float(result[n].sum()) for n in ('plan_cost','increase_cost','refund_amount','reduction_cost',
             'emergency_cost','total_cost','emergency_kwh','surplus_kwh')}
    summary.update(days=len(frames),start_date='2025-02-01',end_date=str(stop.date()),annual_complete=len(frames)==334,
             weight_mode=args.weight_mode,
             initial_soc=float(result.soc_start_kwh.iloc[0]),final_soc=soc,max_validation_error=error,
             decrease_kwh=float(np.maximum(result.plan_kwh-result.final_regular_kwh,0).sum()),
             increase_kwh=float(np.maximum(result.final_regular_kwh-result.plan_kwh,0).sum()),
             emergency_slots=int((result.emergency_kwh>TOL).sum()))
    previous.save_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'results/q3_six_hour_refund')
    parser.add_argument('--end-date',default='2025-12-31')
    parser.add_argument('--time-limit',type=float,default=60)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--newest-only',action='store_true')
    parser.add_argument('--no-adjustments',action='store_true')
    parser.add_argument('--weight-mode',choices=weight_schedule.MODES,default='january',
                        help='january preserves legacy weights; other modes use only mature targets at month start')
    run(parser.parse_args())

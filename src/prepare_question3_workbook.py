"""Prepare official Q3 export only from 334 complete, validated, linked real-data days."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solve_question3 import ROOT, digest, load_inputs, save_json, verify_ledger


def prepare(folder, base=ROOT):
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    df=pd.read_csv(folder/'ledger.csv')
    expected=pd.date_range('2025-02-01','2025-12-31').strftime('%Y-%m-%d').tolist()
    if not summary.get('annual_complete') or df.date.drop_duplicates().tolist()!=expected or len(df)!=334*144:
        raise ValueError('A partial run cannot be exported as official result3.xlsx')
    verify_ledger(df)
    price,load,pv,dates,_=load_inputs(base)
    np.testing.assert_allclose(df.net_kwh,((load[31:]-pv[31:])/6).ravel(),atol=1e-7,rtol=0)
    np.testing.assert_allclose(df.price,np.tile(price,334),atol=1e-10,rtol=0)
    jan=pd.read_csv(folder/'2025-01-31_ledger.csv')
    if abs(jan.soc_end_kwh.iloc[-1]-df.soc_start_kwh.iloc[0])>1e-5:
        raise ValueError('February state does not match January warmup')
    plans,adjustments,batteries,emergencies=[],[],[],[]
    def stamp(slot):
        return f'{slot//6}:{slot%6*10:02d}' if slot<144 else '0:00+1'
    for day,frame in df.groupby('date',sort=False):
        checkpoint=json.loads((folder/f'{day}_complete.json').read_text(encoding='utf-8'))
        if digest(folder/f'{day}_ledger.csv')!=checkpoint['ledger_sha256'] or digest(folder/f'{day}_versions.json')!=checkpoint['versions_sha256']:
            raise ValueError(f'Checkpoint changed: {day}')
        original=pd.read_csv(folder/f'{day}_ledger.csv')
        np.testing.assert_allclose(frame.select_dtypes('number'),original.select_dtypes('number'),atol=1e-8,rtol=0)
        plans.append([day]+frame.plan_kwh.tolist()+[float(frame.plan_kwh.sum()),float(frame.plan_cost.sum())])
        adjustments.append([day]+frame.final_regular_kwh.tolist()+[float(frame.final_regular_kwh.sum()),
                           float(frame[['plan_cost','increase_cost','reduction_cost']].to_numpy().sum())])
        for block in range(6):
            part=frame.iloc[24*block:24*(block+1)]
            batteries.append([day,f'{4*block}:00-{4*(block+1)}:00',float(part.charge_kwh.sum()),float(part.discharge_kwh.sum()),
                              '0:00' if block==0 else '24:00' if block==1 else None,
                              float(frame.soc_start_kwh.iloc[0]) if block==0 else float(frame.soc_end_kwh.iloc[-1]) if block==1 else None])
        quantities=frame.emergency_kwh.to_numpy()
        t=0; segments=[]
        while t<144:
            if quantities[t]<=0:
                t+=1
                continue
            start=t; amount=0.
            while t<144 and quantities[t]>0:
                amount+=quantities[t]; t+=1
            segments.append([day,f'{stamp(start)}-{stamp(t)}',float(amount)])
        emergencies.extend(segments or [[day,'无',0.]])
    return dict(annual_complete=True,labels=[f'{stamp(t)}-{stamp(t+1)}' for t in range(144)],
                plans=plans,adjustments=adjustments,batteries=batteries,emergencies=emergencies)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--base-dir',type=Path,default=ROOT)
    args=parser.parse_args()
    save_json(args.folder/'workbook_payload.json',prepare(args.folder,args.base_dir))

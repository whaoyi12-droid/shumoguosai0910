# -*- coding: utf-8 -*-
"""把全年优化结果写入 result 模板。"""
import numpy as np, openpyxl, datetime as dt
from common import N, PERM
from io_utils import open_template, save, R, block4
BLOCK_LAB=["0:00-4:00","4:00-8:00","8:00-12:00","12:00-16:00","16:00-20:00","20:00-24:00"]
OUT_FROM=dt.date(2025,2,1); OUT_TO=dt.date(2025,12,31)

def _out_index(dates):
    i0=dates.index(OUT_FROM); i1=dates.index(OUT_TO)
    return list(range(i0,i1+1))

def _fill_plan_sheet(ws,res,idxs,key='g',costkey='daycost'):
    g=res[key]; cost=res[costkey]
    for r,i in enumerate(idxs,start=2):
        ws.cell(row=r,column=1,value=res['dates'][i])
        gd=g[i]
        for j in range(144):
            t=PERM[j]
            ws.cell(row=r,column=2+j,value=R(gd[t]))
        ws.cell(row=r,column=146,value=R(gd.sum()))
        ws.cell(row=r,column=147,value=R(cost[i]))

def _rebuild_charge_sheet(wb,res,idxs,sheet="充放电量"):
    del wb[sheet]
    ws=wb.create_sheet(sheet)
    ws.append(["日期","时间段","充电量","放电量","时刻","储电量"])
    rr=2
    for i in idxs:
        b4=block4(res['c'][i],res['d'][i])
        d=res['dates'][i]
        for k in range(6):
            row=[d if k==0 else None, BLOCK_LAB[k], b4[k][0], b4[k][1]]
            if k==0: row+=["0:00",R(res['s_start'][i])]
            elif k==1: row+=["24:00",R(res['s'][i,-1])]
            else: row+=[None,None]
            for cc,v in enumerate(row,start=1): ws.cell(row=rr,column=cc,value=v)
            rr+=1
    return ws

def _rebuild_emerg(wb,emerg_by_date,idxs,dates,sheet="紧急购电量"):
    if sheet in wb.sheetnames: del wb[sheet]
    ws=wb.create_sheet(sheet)
    ws.append(["日期","购电时间段","购电量"])
    rr=2
    for i in idxs:
        recs=emerg_by_date.get(i,[])
        first=True
        for lab,val in recs:
            ws.cell(row=rr,column=1,value=dates[i] if first else None)
            ws.cell(row=rr,column=2,value=lab)
            ws.cell(row=rr,column=3,value=val); rr+=1; first=False
    return ws

def write_q2_like(tpl,outname,res,emerg_by_date=None):
    idxs=_out_index(res['dates'])
    wb=open_template(tpl)
    _fill_plan_sheet(wb["计划购电量"],res,idxs)
    _rebuild_charge_sheet(wb,res,idxs)
    _rebuild_emerg(wb,emerg_by_date or {},idxs,res['dates'])
    # 调整购电量(若模板有且未提供则留空, 问题3单独处理)
    return save(wb,outname)

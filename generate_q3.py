# -*- coding: utf-8 -*-
import numpy as np, datetime as dt, openpyxl
from common import N, PERM
from io_utils import open_template,save,R,block4,merge_emerg
from writer import _out_index, _rebuild_charge_sheet, _rebuild_emerg

def fill(ws,dates,idxs,g,cost):
    for r,i in enumerate(idxs,start=2):
        ws.cell(row=r,column=1,value=dates[i]); gg=g[i]
        for j in range(144): ws.cell(row=r,column=2+j,value=R(gg[PERM[j]]))
        ws.cell(row=r,column=146,value=R(gg.sum())); ws.cell(row=r,column=147,value=R(cost[i]))

def write_result3(tpl,out,B):
    idxs=_out_index(B['dates']); wb=open_template(tpl)
    fill(wb["计划购电量"],B['dates'],idxs,B['gplan'],B['plan'])
    fill(wb["调整购电量"],B['dates'],idxs,B['gfinal'],B['totB'])
    _rebuild_charge_sheet(wb,B,idxs)
    ed={i:merge_emerg(B['emerg'][i]) for i in idxs if B['emerg'][i].sum()>1e-6}
    _rebuild_emerg(wb,ed,idxs,B['dates'])
    return save(wb,out)

if __name__=="__main__":
    from solve_q3 import build
    B=build('fixed')
    p=write_result3("result3.xlsx","result3.xlsx",B)
    wb=openpyxl.load_workbook(p)
    print("saved",p)
    for sh in wb.sheetnames: print(" ",sh,wb[sh].max_row,"x",wb[sh].max_column)
    # 指定日期紧急购电(表3)
    for tgt in [dt.date(2025,3,20),dt.date(2025,6,21),dt.date(2025,9,23),dt.date(2025,12,21)]:
        i=B['dates'].index(tgt); segs=merge_emerg(B['emerg'][i])
        print(tgt,"紧急段数%d 总量%.1f kWh"%(len(segs),B['emerg'][i].sum()), segs[:4])

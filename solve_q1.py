# -*- coding: utf-8 -*-
import numpy as np
from common import *
from optimizer import solve_horizon
from io_utils import *

p1,l1,pv1=load_a1()
r=solve_horizon(l1,pv1,p1,init_free=True,terminal='init')
g,c,d,s,q=[r[k] for k in ('g','c','d','s','q')]
cost=r['cost']; s0=r['s_init']

wb=open_template("result1.xlsx")
ws=wb["计划购电量"]
# 模板行 rr(1..144) <- 内部 t=PERM[rr-1]
for rr in range(1,145):
    t=PERM[rr-1]
    ws.cell(row=rr+1,column=2,value=R(g[t]))
ws2=wb["充放电量"]
b4=block4(c,d)
for k in range(6):
    ws2.cell(row=k+2,column=2,value=b4[k][0])  # 充电量
    ws2.cell(row=k+2,column=3,value=b4[k][1])  # 放电量
ws2.cell(row=2,column=5,value=R(s0))    # 0:00 储电量 (row2 时刻0:00)
ws2.cell(row=3,column=5,value=R(s[-1]))# 24:00 储电量(row3 时刻24:00)
path=save(wb,"result1.xlsx")
print("saved",path)

print("\n===== 表1 指定时间段购电量(kWh) =====")
for lab,t in [("10:00-10:10",60),("12:00-12:10",72),("14:00-14:10",84),
              ("16:00-16:10",96),("18:00-18:10",108),("20:00-20:10",120)]:
    print(f"  {lab}: {R(g[t])}")
print("  全天购电量 %.2f kWh ; 全天购电费 %.2f 元"%(g.sum(),cost))
print("\n===== 表2 储能充放电(kWh) =====")
labs=["0:00-4:00","4:00-8:00","8:00-12:00","12:00-16:00","16:00-20:00","20:00-24:00"]
for k in range(6):
    print(f"  {labs[k]}: 充电 {b4[k][0]}  放电 {b4[k][1]}")
print("  0:00储电量 %.2f ; 24:00储电量 %.2f"%(s0,s[-1]))

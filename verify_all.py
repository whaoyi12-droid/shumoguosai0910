# -*- coding: utf-8 -*-
"""独立回读 outputs 下结果文件做校验(不依赖求解期内存)。"""
import numpy as np, pandas as pd, os, datetime as dt
from common import N, PERM, CAP_MIN, CAP_MAX, load_a1, load_a2, load_a4
OUT=os.path.join(os.path.dirname(__file__),"outputs")
def rd(f,sh): return pd.read_excel(os.path.join(OUT,f),sheet_name=sh,header=0)

print("="*70,"\n[result1]")
g=rd("result1.xlsx","计划购电量"); print("购电量行",len(g),"列",list(g.columns))
vals=g.iloc[:,1].to_numpy(float)
print(" 全天购电量=%.2f (应≈59482.70), 无负值%d"%(np.nansum(vals),(vals>=-1e-6).all()))
cd=rd("result1.xlsx","充放电量"); print(cd.to_string(index=False))

def check_year(f,has_adj):
    print("="*70,"\n[%s]"%f)
    plan=rd(f,"计划购电量"); nd=len(plan)
    seg=plan.iloc[:,1:145].to_numpy(float); tot=plan.iloc[:,145].to_numpy(float); fee=plan.iloc[:,146].to_numpy(float)
    err=np.abs(seg.sum(1)-tot).max(); print(" 计划 %d天, 行和vs全天列 最大误差 %.4f, 购电非负%s"%(nd,err,(seg>=-1e-6).all()))
    ch=rd(f,"充放电量"); print(" 充放电行数%d (应%d)"%(len(ch),nd*6))
    soc=pd.to_numeric(ch.iloc[:,5],errors='coerce').dropna().to_numpy()
    print(" SOC[%.1f,%.1f] 越界%s"%(soc.min(),soc.max(),((soc<CAP_MIN-1e-4)|(soc>CAP_MAX+1e-4)).any()))
    em=rd(f,"紧急购电量"); print(" 紧急记录行数",len(em))
    if has_adj:
        adj=rd(f,"调整购电量"); a=adj.iloc[:,1:145].to_numpy(float)
        print(" 调整 %d天, 非负%s, 调整全天列和%.1f"%(len(a),(a>=-1e-6).all(),adj.iloc[:,145].sum()))
    # 电费校验(固定电价用附件1)
    return plan

p2=check_year("result2.xlsx",False)
p3=check_year("result3.xlsx",True)
p42=check_year("result4-2.xlsx",False)
p43=check_year("result4-3.xlsx",True)

# 电费独立重算校验 result2 (固定电价)
p1,_,_=load_a1()
seg=p2.iloc[:,1:145].to_numpy(float)
# 列顺序是PERM, 还原到自然t: 列j对应t=PERM[j]
nat=np.zeros_like(seg)
for j in range(144): nat[:,PERM[j]]=seg[:,j]
fee_calc=(nat*p1[None,:]).sum(1)
fee_tab=p2.iloc[:,146].to_numpy(float)
print("\nresult2 电费独立重算 vs 表内 最大偏差 %.4f 元"%np.abs(fee_calc-fee_tab).max())
print("result2 输出首日",p2.iloc[0,0],"末日",p2.iloc[-1,0])

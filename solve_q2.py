# -*- coding: utf-8 -*-
import time, numpy as np, datetime as dt
import openpyxl
from common import *
from optimizer import solve_horizon
from io_utils import *

def run(price_mode='fixed', tag="q2"):
    """price_mode: 'fixed'=附件1电价每日相同; 'volatile'=附件4波动电价。返回全年结果与写出。"""
    p1,l1,pv1=load_a1()
    dL,L,P=load_a2()
    if price_mode=='fixed':
        PRICE=np.tile(p1,(len(dL),1))           # (365,144)
    else:
        d4,PRICE=load_a4(); assert d4==dL
    t0=time.time()
    r=solve_horizon(L,P,PRICE,s_init=SOC0,init_free=False,terminal='value',term_val=SOC0)
    print("[%s] 求解用时 %.1f s, 状态%d"%(tag,time.time()-t0,r['status']))
    M=L.size
    g=r['g'].reshape(-1,N); c=r['c'].reshape(-1,N); dd=r['d'].reshape(-1,N)
    s=r['s'].reshape(-1,N); q=r['q'].reshape(-1,N)
    daycost=(PRICE*g).sum(axis=1)
    print("全年购电费 %.2f 元, 全年购电量 %.0f kWh, SOC[%.1f,%.1f], 弃光 %.2f kWh"%(
        daycost.sum(),g.sum(),s.min(),s.max(),q.sum()))
    # 日初SOC: 1-1=6000, 其后=前日末
    s_start=np.r_[[SOC0],s[:-1,-1]]
    return dict(dates=dL,L=L,P=P,PRICE=PRICE,g=g,c=c,d=dd,s=s,q=q,daycost=daycost,
                s_start=s_start)

if __name__=="__main__":
    res=run('fixed','q2')
    # 指定日期表3(问题2紧急购电=0, 因确定性)。检查执行平衡: 计划即基于实际 -> 残差
    bal=res['g']+res['d']-res['c']-res['q']-(res['L']*DT-res['P']*DT)
    print("功率平衡最大残差 %.2e"%np.abs(bal).max())
    print("输出区间 2-1 idx=%d, 12-31 idx=%d, 天数=%d"%(
        res['dates'].index(dt.date(2025,2,1)),res['dates'].index(dt.date(2025,12,31)),
        res['dates'].index(dt.date(2025,12,31))-res['dates'].index(dt.date(2025,2,1))+1))
    for tgt in [dt.date(2025,3,20),dt.date(2025,6,21),dt.date(2025,9,23),dt.date(2025,12,21)]:
        i=res['dates'].index(tgt)
        print(tgt,"购电量%.1f 购电费%.1f 日初SOC%.1f 日末SOC%.1f"%(
            res['g'][i].sum(),res['daycost'][i],res['s_start'][i],res['s'][i,-1]))

# -*- coding: utf-8 -*-
"""试验不同预报融合方式对全年总费用的影响, 决定问题3最优调整策略。"""
import numpy as np, datetime as dt, time
from common import *
from optimizer import solve_horizon
from mpc import solve_window

def mpc_blend(load,pv,price,fc_all,sstart,send,mode):
    gplan=np.zeros(N);gf=np.zeros(N);cf=np.zeros(N);df=np.zeros(N);sf=np.zeros(N);em=np.zeros(N)
    def use_fc(h):
        if h==0: return fc_all[0]
        if mode=='latest': return fc_all[h]
        if mode=='min': return np.minimum(fc_all[0],fc_all[h])
        if mode=='avg': return 0.5*(fc_all[0]+fc_all[h])
    # 0点全天
    s0=solve_window(load,fc_all[0],price,0,N,sstart,send)
    gplan[:]=s0['g']
    def exec_(a,b,sol):
        gg=sol['g'][:b-a];cc=sol['c'][:b-a];dd=sol['d'][:b-a]
        gf[a:b]=gg;cf[a:b]=cc;df[a:b]=dd;sf[a:b]=sol['s'][:b-a]
        em[a:b]=np.maximum(load[a:b]*DT+cc-gg-dd-pv[a:b]*DT,0)
        return sol['s'][b-a-1]
    sc=exec_(0,36,s0)
    for a,b in [(36,72),(72,108),(108,144)]:
        sol=solve_window(load,use_fc(a//6),price,a,b,sc,send,g_plan=gplan)
        sc=exec_(a,b,sol)
    tot=(price*gf).sum()+(0.5*price*np.abs(gf-gplan)).sum()+(5*price*em).sum()
    return tot,em.sum()

def run():
    p1,_,_=load_a1();dL,L,P=load_a2();a3=load_a3();PRICE=np.tile(p1,(len(dL),1))
    ref=solve_horizon(L,P,PRICE,terminal='value',term_val=SOC0)['s'].reshape(-1,N)
    sstart=np.r_[[SOC0],ref[:-1,-1]];i0=dL.index(dt.date(2025,2,1))
    i0=dL.index(dt.date(2025,2,1))
    res={m:0.0 for m in ['latest','min','avg']}; emm={m:0.0 for m in res}
    for i in range(i0,len(dL)):
        fc={h:fc_to_10min(a3[dL[i]][h],h) for h in [0,6,12,18]}
        for m in res:
            t,e=mpc_blend(L[i],P[i],PRICE[i],fc,sstart[i],ref[i,-1],m)
            res[m]+=t;emm[m]+=e
    for m in res: print("模式 %-7s 总费用 %.2f 元, 紧急 %.0f kWh"%(m,res[m],emm[m]))

if __name__=="__main__":
    t=time.time();run();print("用时%.1fs"%(time.time()-t))

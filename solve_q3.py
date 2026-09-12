# -*- coding: utf-8 -*-
import numpy as np, datetime as dt, time
from common import *
from optimizer import solve_horizon
from mpc import solve_window, mpc_day

def no_adjust_day(load,pv,price,pv_fc0,s_start,s_end):
    """方案A: 只用0点预报定全天计划, 之后不调整, 按实际核算紧急。"""
    sol=solve_window(load,pv_fc0,price,0,N,s_start,s_end,g_plan=None)
    g,c,d,s=sol['g'],sol['c'],sol['d'],sol['s']
    need=load*DT+c-g-d-pv*DT
    emerg=np.maximum(need,0)
    total=float((price*g).sum()+(5*price*emerg).sum())
    return dict(g=g,c=c,d=d,s=s,emerg=emerg,total=total)

def build(price_mode='fixed'):
    p1,_,_=load_a1(); dL,L,P=load_a2(); a3=load_a3()
    if price_mode=='fixed': PRICE=np.tile(p1,(len(dL),1))
    else:
        d4,PRICE=load_a4(); assert d4==dL
    # 参考跨天SOC轨迹: 全局确定性经济调度
    ref=solve_horizon(L,P,PRICE,s_init=SOC0,terminal='value',term_val=SOC0)
    sref=ref['s'].reshape(-1,N); sstart=np.r_[[SOC0],sref[:-1,-1]]
    nd=len(dL)
    def alloc():
        return dict(gplan=np.zeros((nd,N)),gfinal=np.zeros((nd,N)),c=np.zeros((nd,N)),
                    d=np.zeros((nd,N)),s=np.zeros((nd,N)),emerg=np.zeros((nd,N)),
                    plan=np.zeros(nd),adj=np.zeros(nd),emc=np.zeros(nd),
                    final_e=np.zeros(nd),totB=np.zeros(nd),totA=np.zeros(nd))
    B=alloc()
    B['emA']=np.zeros(nd); B['enA']=np.zeros(nd); B['totC']=np.zeros(nd)
    t0=time.time()
    for i in range(nd):
        fc={h:fc_to_10min(a3[dL[i]][h],h) for h in [0,6,12,18]}
        rb=mpc_day(L[i],P[i],PRICE[i],fc,sstart[i],sref[i,-1])
        B['gplan'][i]=rb['g_plan'];B['gfinal'][i]=rb['g'];B['c'][i]=rb['c'];B['d'][i]=rb['d']
        B['s'][i]=rb['s'];B['emerg'][i]=rb['emerg']
        B['plan'][i]=rb['plan_cost'];B['adj'][i]=rb['adj_cost'];B['emc'][i]=rb['emerg_cost']
        B['final_e'][i]=rb['final_energy_cost'];B['totB'][i]=rb['total']
        ra=no_adjust_day(L[i],P[i],PRICE[i],fc[0],sstart[i],sref[i,-1])
        B['totA'][i]=ra['total']; B['emA'][i]=ra['emerg'].sum()
        B['enA'][i]=(PRICE[i]*ra['g']).sum()
        # 方案C: 每6小时用"实际"光伏调整(完美预见), 给出调整价值上界
        fcc={0:fc[0],6:P[i],12:P[i],18:P[i]}
        rc=mpc_day(L[i],P[i],PRICE[i],fcc,sstart[i],sref[i,-1])
        B['totC'][i]=rc['total']
    B.update(dates=dL,L=L,P=P,PRICE=PRICE,s_start=sstart,s_end=sref[:,-1])
    print("[%s] 全年MPC用时 %.1fs"%(price_mode,time.time()-t0))
    return B

if __name__=="__main__":
    B=build('fixed')
    i0=B['dates'].index(dt.date(2025,2,1))
    sl=slice(i0,None)
    print("=== 2.1-12.31 汇总(固定电价) ===")
    print("方案A 只用0点预报 总费用 %.2f 元 (紧急%.0f kWh, 紧急费%.0f, 能源费%.0f)"%(
        B['totA'][sl].sum(),B['emA'][sl].sum(),
        B['totA'][sl].sum()-B['enA'][sl].sum(),B['enA'][sl].sum()))
    print("方案B 滚动调整    总费用 %.2f 元 (紧急%.0f kWh)"%(B['totB'][sl].sum(),B['emerg'][sl].sum()))
    print("  其中 最终能源费 %.2f, 调整摩擦费 %.2f, 紧急购电费 %.2f"%(
        B['final_e'][sl].sum(),B['adj'][sl].sum(),B['emc'][sl].sum()))
    print("方案C 完美预见调整 总费用 %.2f 元"%B['totC'][sl].sum())
    print("  B-A = %.2f 元 (%.2f%%);  C-A = %.2f 元 (%.2f%%)"%(
        B['totB'][sl].sum()-B['totA'][sl].sum(),
        100*(B['totB'][sl].sum()-B['totA'][sl].sum())/B['totA'][sl].sum(),
        B['totC'][sl].sum()-B['totA'][sl].sum(),
        100*(B['totC'][sl].sum()-B['totA'][sl].sum())/B['totA'][sl].sum()))
    emA_tot=0
    for tgt in [dt.date(2025,3,20),dt.date(2025,6,21),dt.date(2025,9,23),dt.date(2025,12,21)]:
        i=B['dates'].index(tgt)
        print("%s: 计划购电%.0f 最终购电%.0f 紧急%.1f kWh 总费%.1f"%(
            tgt,B['gplan'][i].sum(),B['gfinal'][i].sum(),B['emerg'][i].sum(),B['totB'][i]))

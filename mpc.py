# -*- coding: utf-8 -*-
"""问题3/4-3 滚动时域(MPC)窗口优化器。
在窗口[a,b)内, 已知初始SOC(sa)与终端目标(sb), 用(预报光伏,实际负荷)优化。
若提供 g_plan 参考, 线性化调整摩擦: 目标含 0.5*price*|g-g_plan| (u-v分解)。"""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from common import *

def solve_window(load_kW, pv_kW, price, a, b, sa, sb, g_plan=None):
    L=np.asarray(load_kW,float); P=np.asarray(pv_kW,float); pr=np.asarray(price,float)
    m=b-a
    D=L[a:b]*DT; EE=P[a:b]*DT; pp=pr[a:b]
    has_adj = g_plan is not None
    # 变量块: g,c,d,s,q (=5m), 可选 u,v
    base=5*m; nv=base+(2*m if has_adj else 0)
    def G(t):return t
    def C(t):return m+t
    def Dd(t):return 2*m+t
    def S(t):return 3*m+t
    def Q(t):return 4*m+t
    def U(t):return base+t
    def V(t):return base+m+t
    rows=[];cols=[];vals=[];beq=[]
    for t in range(m):
        r=t
        rows+=[r,r,r,r];cols+=[G(t),Dd(t),C(t),Q(t)];vals+=[1,1,-1,-1];beq.append(D[t]-EE[t])
    for t in range(m):
        r=m+t
        rows+=[r,r,r];cols+=[S(t),C(t),Dd(t)];vals+=[1,-ETA,1/ETA]
        if t==0: beq.append(sa)
        else: rows.append(r);cols.append(S(t-1));vals.append(-1);beq.append(0.0)
    neq=2*m
    # 终端SOC
    rows.append(neq);cols.append(S(m-1));vals.append(1);beq.append(sb);neq+=1
    if has_adj:  # g - u + v = g_plan
        gp=np.asarray(g_plan,float)[a:b]
        for t in range(m):
            rows+=[neq,neq,neq];cols+=[G(t),U(t),V(t)];vals+=[1,-1,1];beq.append(gp[t]);neq+=1
    A=sp.coo_matrix((vals,(rows,cols)),shape=(neq,nv)).tocsr(); bv=np.array(beq,float)
    cobj=np.zeros(nv); cobj[:m]=pp
    if has_adj: cobj[base:base+2*m]=np.r_[0.5*pp,0.5*pp]
    lb=np.full(nv,-np.inf);ub=np.full(nv,np.inf)
    lb[:m]=0; lb[m:2*m]=0;ub[m:2*m]=SEG_E
    lb[2*m:3*m]=0;ub[2*m:3*m]=SEG_E
    lb[3*m:4*m]=CAP_MIN;ub[3*m:4*m]=CAP_MAX
    lb[4*m:5*m]=0
    if has_adj: lb[base:base+2*m]=0
    res=linprog(cobj,A_eq=A,b_eq=bv,bounds=list(zip(lb,ub)),method='highs')
    if not res.success: raise RuntimeError("window LP失败 a=%d b=%d %s"%(a,b,res.message))
    x=res.x
    return dict(g=x[:m],c=x[m:2*m],d=x[2*m:3*m],s=x[3*m:4*m],q=x[4*m:5*m])

def mpc_day(load, pv, price, pv_fc, s_start, s_end, blend='min'):
    """单日MPC。pv_fc: dict {0,6,12,18: 144预报功率}。
    blend: 日内更新预报与0点预报的融合方式('latest'/'min'/'avg');
    风险厌恶下取min(宁可少估光伏、多购电, 以1.5倍代价规避5倍紧急购电)。"""
    def blended(h):
        f0,fh=pv_fc[0],pv_fc[h]
        if blend=='latest': return fh
        if blend=='avg': return 0.5*(f0+fh)
        return np.minimum(f0,fh)
    g_plan=np.zeros(N); c_plan=np.zeros(N);d_plan=np.zeros(N);s_plan=np.zeros(N)
    g_f=np.zeros(N);c_f=np.zeros(N);d_f=np.zeros(N);s_f=np.zeros(N);emerg=np.zeros(N)
    def execute(a,b,sol):
        g_f[a:b]=sol['g'][a-b:] if False else sol['g'][:b-a]
        c_f[a:b]=sol['c'][:b-a]; d_f[a:b]=sol['d'][:b-a]; s_f[a:b]=sol['s'][:b-a]
        D=load[a:b]*DT; Pact=pv[a:b]*DT
        gg=sol['g'][:b-a];cc=sol['c'][:b-a];dd=sol['d'][:b-a]
        need=D+cc-gg-dd-Pact                      # >0 供电不足->紧急购电
        emerg[a:b]=np.maximum(need,0)
        return sol['s'][b-a-1]
    # 0点: 用0点预报一次性制定全天计划(窗口0-144), 执行[0,36)
    sol0=solve_window(load,pv_fc[0],price,0,N,s_start,s_end,g_plan=None)
    g_plan[:]=sol0['g'];c_plan[:]=sol0['c'];d_plan[:]=sol0['d'];s_plan[:]=sol0['s']
    scur=execute(0,36,sol0)
    # 6/12/18点: 用最新预报重优化剩余窗口, 只执行到下个预报点
    for a,b in [(36,72),(72,108),(108,144)]:
        sol=solve_window(load,blended(a//6),price,a,b,scur,s_end,g_plan=g_plan)
        scur=execute(a,b,sol)
    # 费用核算(对称摩擦形式)
    plan_cost=(price*g_plan).sum()*DT/ DT  # 计划购电费用
    plan_cost=float((price*g_plan).sum())
    delta=np.abs(g_f-g_plan)
    adj_cost=float((0.5*price*delta).sum())
    emerg_cost=float((5*price*emerg).sum())
    base_final=float((price*g_f).sum())
    total=base_final+adj_cost+emerg_cost
    return dict(g_plan=g_plan,c_plan=c_plan,d_plan=d_plan,s_plan=s_plan,
                g=g_f,c=c_f,d=d_f,s=s_f,emerg=emerg,
                plan_cost=plan_cost,adj_cost=adj_cost,emerg_cost=emerg_cost,
                final_energy_cost=base_final,total=total)

if __name__=="__main__":
    import datetime as dt
    from common import load_a2,load_a3,fc_to_10min,load_a1
    p1,l1,pv1=load_a1(); dL,L,P=load_a2(); a3=load_a3()
    # 用固定电价(附件1), 测试 3-20 (idx=78)
    tgt=dt.date(2025,3,20); i=dL.index(tgt)
    fc={h:fc_to_10min(a3[tgt][h],h) for h in [0,6,12,18]}
    r=mpc_day(L[i],P[i],p1,fc,6000,6000)
    print("3-20 MPC: 计划购电%.1f 最终购电%.1f 紧急%.2f kWh"%(r['g_plan'].sum(),r['g'].sum(),r['emerg'].sum()))
    print(" 计划费%.1f 调整费%.1f 紧急费%.1f 最终电费%.1f 合计%.1f"%(
        r['plan_cost'],r['adj_cost'],r['emerg_cost'],r['final_energy_cost'],r['total']))
    print(" 紧急段数",(r['emerg']>1e-6).sum()," SOC[%.0f,%.0f] 末%.0f"%(r['s'].min(),r['s'].max(),r['s'][-1]))

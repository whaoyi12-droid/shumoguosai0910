# -*- coding: utf-8 -*-
"""核心线性规划求解器。
把若干天串成 M=T*144 段连续序列, SOC 全程递推(天然跨天衔接)。
变量(每段): g购电, c充电, d放电, s段末SOC, q弃光; 单位kWh。
功率平衡(交流侧): g + d - c - q = D-P   (D=负荷电量,P=光伏发电量)
SOC递推(电池侧):   s[t]-s[t-1] = eta*c[t] - d[t]/eta
"""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from common import *

def solve_horizon(load_kW, pv_kW, price, s_init=SOC0,
                  init_free=False, terminal='value', term_val=SOC0,
                  emerg_kW=None, verbose=False):
    """load_kW,pv_kW,price: shape (M,) 或 (T,144) 的扁平序列(功率kW / 电价)。
    init_free: 首段初始SOC是否为自由变量(问题1)。
    terminal: 'value'=末端SOC=term_val; 'init'=末端=初始(自由初值时); None=自由。
    返回 dict: g,c,d,s,q (M,), s_init_val, cost(按price·g), status"""
    L=np.asarray(load_kW,float).reshape(-1); P=np.asarray(pv_kW,float).reshape(-1)
    pr=np.asarray(price,float).reshape(-1); M=L.size
    assert P.size==M and pr.size==M
    D=L*DT; EE=P*DT
    nbase=5*M
    has_x0=bool(init_free)
    nv=nbase+(1 if has_x0 else 0)
    def G(t):return t
    def C(t):return M+t
    def Dd(t):return 2*M+t
    def S(t):return 3*M+t
    def Q(t):return 4*M+t
    rows=[];cols=[];vals=[];beq=[]
    # --- 功率平衡 M 条 ---
    for t in range(M):
        r=t
        rows+=[r,r,r,r]; cols+=[G(t),Dd(t),C(t),Q(t)]; vals+=[1,1,-1,-1]
        beq.append(D[t]-EE[t])
    # --- SOC 递推 M 条 ---
    for t in range(M):
        r=M+t
        rows.append(r);cols.append(S(t));vals.append(1)
        rows.append(r);cols.append(C(t));vals.append(-ETA)
        rows.append(r);cols.append(Dd(t));vals.append(1/ETA)
        if t==0:
            if has_x0:
                rows.append(r);cols.append(nbase);vals.append(-1); beq.append(0.0)
            else:
                beq.append(s_init)
        else:
            rows.append(r);cols.append(S(t-1));vals.append(-1); beq.append(0.0)
    n_eq=2*M
    # --- 终端约束 ---
    if terminal=='value':
        rows.append(n_eq);cols.append(S(M-1));vals.append(1);beq.append(term_val);n_eq+=1
    elif terminal=='init' and has_x0:
        rows.append(n_eq);cols.append(S(M-1));vals.append(1)
        rows.append(n_eq);cols.append(nbase);vals.append(-1);beq.append(0.0);n_eq+=1
    A=sp.coo_matrix((vals,(rows,cols)),shape=(n_eq,nv)).tocsr()
    b=np.array(beq,float)
    # 目标: 仅购电有费用
    cobj=np.zeros(nv); cobj[:M]=pr
    # 界
    lb=np.full(nv,-np.inf); ub=np.full(nv,np.inf)
    lb[:M]=0; ub[:M]=np.inf                    # g
    lb[M:2*M]=0; ub[M:2*M]=SEG_E               # c
    lb[2*M:3*M]=0; ub[2*M:3*M]=SEG_E           # d
    lb[3*M:4*M]=CAP_MIN; ub[3*M:4*M]=CAP_MAX   # s
    lb[4*M:5*M]=0; ub[4*M:5*M]=np.inf          # q
    bounds=list(zip(lb,ub))
    if has_x0: bounds[nbase]=(CAP_MIN,CAP_MAX)
    res=linprog(cobj,A_eq=A,b_eq=b,bounds=bounds,method='highs',
                options={'presolve':True})
    if not res.success:
        raise RuntimeError("LP失败: "+res.message)
    x=res.x
    out=dict(g=x[:M].copy(),c=x[M:2*M].copy(),d=x[2*M:3*M].copy(),
             s=x[3*M:4*M].copy(),q=x[4*M:5*M].copy(),
             cost=float((pr*x[:M]).sum()),status=res.status)
    out['s_init']=float(x[nbase]) if has_x0 else float(s_init)
    return out

if __name__=="__main__":
    p1,l1,pv1=load_a1()
    r=solve_horizon(l1,pv1,p1,init_free=True,terminal='init')
    g,c,d,s,q=(r[k] for k in ('g','c','d','s','q'))
    print("状态",r['status']," 购电费 %.2f 元"%r['cost'])
    print("全天购电量 %.2f kWh ; 负荷电量 %.2f ; 光伏发电量 %.2f"%(g.sum(),(l1*DT).sum(),(pv1*DT).sum()))
    print("初始/末端SOC %.2f / %.2f ; SOC范围[%.1f,%.1f]"%(r['s_init'],s[-1],s.min(),s.max()))
    print("充电合计%.2f 放电合计%.2f 弃光%.2f"%(c.sum(),d.sum(),q.sum()))
    print("c上限触界段数%d d上限触界%d"%((c>SEG_E-1e-6).sum(),(d>SEG_E-1e-6).sum()))
    # 平衡残差
    res=(g+d-c-q-(l1*DT-pv1*DT))
    print("平衡最大残差 %.2e"%np.abs(res).max())

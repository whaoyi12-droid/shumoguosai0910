# -*- coding: utf-8 -*-
"""统一数据加载层 + 物理常量。内部时段顺序 t=0..143 对应 00:00-00:10 ... 23:50-24:00。"""
import pandas as pd, numpy as np, os, datetime as dt
DESK = r"C:\Users\m4113\Desktop"
N = 144                 # 每天144段(10min)
DT = 10/60              # 段长 h = 1/6
ETA = 0.9               # 充放电效率(充、放各一次)
CAP_MAX, CAP_MIN = 10800.0, 1200.0     # SOC上下限 kWh
PMAX = 5000.0                            # 最大充放电功率 kW
SEG_E = PMAX*DT                          # 每段最大充/放电量 kWh = 833.333
SOC0 = 6000.0                            # 2025-01-01 0:00 电量
# 模板购电量列顺序对应内部 t: 列1..143 <- t1..143, 列144 <- t0
PERM = list(range(1,144))+[0]

def _datekey(x):
    s=str(x).strip()
    return pd.to_datetime(s).date()

def load_a1():
    df=pd.read_excel(os.path.join(DESK,"附件1.xlsx"))
    df.columns=[str(c).strip() for c in df.columns]
    price=df.iloc[:,1].to_numpy(float)
    load =df.iloc[:,2].to_numpy(float)
    pv   =df.iloc[:,3].to_numpy(float)
    return price,load,pv

def _sheet_matrix(fname,sheet):
    df=pd.read_excel(os.path.join(DESK,fname),sheet_name=sheet,header=0)
    dates=[_datekey(x) for x in df.iloc[:,0]]
    arr=df.iloc[:,1:1+N].to_numpy(float)   # 365 x 144
    return dates,arr

def load_a2():
    dL,L=_sheet_matrix("附件2.xlsx","小区负载")
    dP,P=_sheet_matrix("附件2.xlsx","光伏发电实际功率")
    assert dL==dP, "附件2两sheet日期不一致"
    return dL,L,P                       # dates, load(365,144), pv(365,144) 功率kW

def load_a4():
    d,price=_sheet_matrix("附件4.xlsx",0)
    return d,price                     # dates,(365,144) 元/kWh

def load_a3():
    """返回 {date:{issue_hour(0/6/12/18): np.array[24] 整点功率, h_k->issue+k点钟}}"""
    df=pd.read_excel(os.path.join(DESK,"附件3.xlsx"),header=0)
    df.columns=[str(c).strip() for c in df.columns]
    datecol=pd.to_datetime(df.iloc[:,0].ffill())   # 合并单元格: 日期向下填充
    out={}
    for i in range(df.shape[0]):
        d=datecol.iloc[i].date()
        ih=int(str(df.iloc[i,1]).split(':')[0])
        fc=df.iloc[i,2:26].to_numpy(float)
        out.setdefault(d,{})[ih]=fc
    return out

def fc_to_10min(fc,issue_hour):
    """把某次预报(整点h1..h24 = issue+1..issue+24点钟)映射到当天 t=0..143 的功率。
    整点值 P[hour], 对10min段(用段末刻,即10,20,..分)在相邻整点间线性插值; 落在预报窗外按0。"""
    # 建立全局钟点 0..48 的值: hour h 对应 fc 中 (h-issue) 索引(1..24)->idx h-issue-1
    def hourval(h):
        k=h-issue_hour           # =1..24 有效
        if 1<=k<=24: return fc[k-1]
        return 0.0
    out=np.zeros(N)
    for t in range(N):
        end_min=(t+1)*10         # 段末分钟 10..1440
        h_float=end_min/60.0     # 0.1667..24
        h0=int(np.floor(h_float)); frac=h_float-h0
        v0=hourval(h0); v1=hourval(h0+1)
        out[t]=v0+(v1-v0)*frac
    return out

if __name__=="__main__":
    p1,l1,pv1=load_a1()
    print("附件1:",p1.shape,"电价[%.4f,%.4f] 负载[%.1f,%.1f] 光伏[%.1f,%.1f]"%(
        p1.min(),p1.max(),l1.min(),l1.max(),pv1.min(),pv1.max()))
    d,L,P=load_a2()
    print("附件2: 天数",len(d),d[0],d[-1],"负载NaN",np.isnan(L).sum(),"光伏NaN",np.isnan(P).sum(),
          "负载范围[%.0f,%.0f]"%(L.min(),L.max()),"光伏[%.0f,%.0f]"%(P.min(),P.max()))
    d4,pr4=load_a4()
    print("附件4:",len(d4),d4[0],d4[-1],"电价[%.4f,%.4f] NaN%d"%(pr4.min(),pr4.max(),np.isnan(pr4).sum()))
    a3=load_a3()
    print("附件3: 天数",len(a3),"示例1-1 issues",list(a3[dt.date(2025,1,1)].keys()))
    # 校验: 0点预报插值后白天能量 vs 24整点
    fc= a3[dt.date(2025,1,1)][0]
    v=fc_to_10min(fc,0)
    print("  0点预报 插值10min 白天峰值%.1f, 总发电≈%.0f kWh; 整点梯形积分≈%.0f kWh"%(
        v.max(),(v*DT).sum(),np.trapezoid(np.r_[0,fc],dx=1)))
    print("  t=42(07:00-07:10末刻7.167)=%.1f  t=48(08:00末)=%.1f"%(v[42],v[47]))

# -*- coding: utf-8 -*-
import numpy as np, datetime as dt, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
from common import *
from optimizer import solve_horizon
from solve_q2 import run as run_q2
from solve_q3 import build as build_q3
from io_utils import block4, merge_emerg, R
OUT=os.path.join(os.path.dirname(__file__),"outputs"); FIG=os.path.join(OUT,"figs"); os.makedirs(FIG,exist_ok=True)
SPEC_T=[("10:00-10:10",60),("12:00-12:10",72),("14:00-14:10",84),("16:00-16:10",96),("18:00-18:10",108),("20:00-20:10",120)]
DATES=[dt.date(2025,3,20),dt.date(2025,6,21),dt.date(2025,9,23),dt.date(2025,12,21)]
S={}

# ---------- 问题1 ----------
p1,l1,pv1=load_a1(); q1=solve_horizon(l1,pv1,p1,init_free=True,terminal='init')
S['q1']={'spec_g':{lab:R(q1['g'][t]) for lab,t in SPEC_T},
         'day_g':R(q1['g'].sum()),'cost':R(q1['cost']),'s0':R(q1['s_init']),'s24':R(q1['s'][-1]),
         'block':[{'ch':a,'dis':b} for a,b in block4(q1['c'],q1['d'])]}
# 图1 问题1日内
    # 时间轴
hours=np.arange(144)/6.0
fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True)
ax[0].plot(hours,l1*DT*6,color='gray',label='负荷功率'); ax[0].plot(hours,pv1,color='orange',label='光伏功率'); ax[0].set_ylabel('kW');ax[0].legend(loc='upper left');ax[0].set_title('问题1 负荷与光伏')
ax[1].plot(hours,q1['g']/DT,color='blue',label='购电功率');ax[1].plot(hours,q1['c']/DT,color='green',label='充电功率');ax[1].plot(hours,q1['d']/DT,color='red',label='放电功率');ax[1].set_ylabel('kW');ax[1].legend(loc='upper left');ax[1].set_title('购电与储能充放电功率')
ax2=ax[1].twinx();ax2.plot(hours,p1,'--',color='purple',lw=1,label='电价');ax2.set_ylabel('元/kWh',color='purple')
soc=np.r_[q1['s_init'],q1['s']];ax[2].plot(np.arange(145)/6,soc,color='black');ax[2].axhline(1200,ls=':',color='gray');ax[2].axhline(10800,ls=':',color='gray');ax[2].set_ylabel('SOC kWh');ax[2].set_xlabel('时');ax[2].set_title('储能SOC');ax[2].set_xlim(0,24)
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig1_q1_day.png'),dpi=130);plt.close()

# ---------- 问题2 固定 ----------
q2=run_q2('fixed','q2')
# ---------- 问题3 固定 ----------
q3=build_q3('fixed')
# ---------- 问题4 ----------
q42=run_q2('volatile','q4-2')
q43=build_q3('volatile')

def year_sum(res,sl_key='totB'):
    i0=res['dates'].index(dt.date(2025,2,1));sl=slice(i0,None);return sl
sl=year_sum(q3)
S['year_fixed']={'q2_cost':R(q2['daycost'][sl].sum()),
  'q3_A':R(q3['totA'][sl].sum()),'q3_B':R(q3['totB'][sl].sum()),'q3_C':R(q3['totC'][sl].sum()),
  'q3_plan':R(q3['plan'][sl].sum()),'q3_adj':R(q3['adj'][sl].sum()),'q3_emergcost':R(q3['emc'][sl].sum()),
  'q3_emerg_kwh_A':R(q3['emA'][sl].sum()),'q3_emerg_kwh_B':R(q3['emerg'][sl].sum())}
sl4=year_sum(q43)
S['year_volatile']={'q42_cost':R(q42['daycost'][sl4].sum()),
  'q43_A':R(q43['totA'][sl4].sum()),'q43_B':R(q43['totB'][sl4].sum()),'q43_C':R(q43['totC'][sl4].sum()),
  'q43_emerg_kwh_B':R(q43['emerg'][sl4].sum())}

# 指定日期表1/2/3
def day_tables(res,res43):
    out={}
    for tgt in DATES:
        i=res['dates'].index(tgt); d={}
        d['spec_g']={lab:R(res['g' if 'g' in res else 'gfinal'][i][t]) for lab,t in SPEC_T} if False else None
        out[str(tgt)]=i
    return out
# 问题2/3 指定日
S['days_fixed']={}
for tgt in DATES:
    i=q3['dates'].index(tgt)
    S['days_fixed'][str(tgt)]={
      'q2_g_spec':{lab:R(q2['g'][i][t]) for lab,t in SPEC_T},'q2_dayg':R(q2['g'][i].sum()),'q2_fee':R(q2['daycost'][i]),
      'q2_block':[{'ch':a,'dis':b} for a,b in block4(q2['c'][i],q2['d'][i])],'q2_s0':R(q2['s_start'][i]),'q2_s24':R(q2['s'][i,-1]),
      'q3_plan_spec':{lab:R(q3['gplan'][i][t]) for lab,t in SPEC_T},'q3_adj_spec':{lab:R(q3['gfinal'][i][t]) for lab,t in SPEC_T},
      'q3_plan_dayg':R(q3['gplan'][i].sum()),'q3_final_dayg':R(q3['gfinal'][i].sum()),'q3_total':R(q3['totB'][i]),
      'q3_block':[{'ch':a,'dis':b} for a,b in block4(q3['c'][i],q3['d'][i])],'q3_s0':R(q3['s_start'][i]),'q3_s24':R(q3['s'][i,-1]),
      'q3_emerg_segs':merge_emerg(q3['emerg'][i]),'q3_emerg_kwh':R(q3['emerg'][i].sum()),
      'q43_emerg_segs':merge_emerg(q43['emerg'][i]),'q43_emerg_kwh':R(q43['emerg'][i].sum()),
      'q42_g_spec':{lab:R(q42['g'][i][t]) for lab,t in SPEC_T},'q42_fee':R(q42['daycost'][i]),
      'q43_total':R(q43['totB'][i])}

# 图2 方案费用对比
fig,ax=plt.subplots(1,2,figsize=(11,4))
yf=S['year_fixed']
ax[0].bar(['只0点(A)','保守调整(B)','完美预见(C)'],[yf['q3_A'],yf['q3_B'],yf['q3_C']],color=['#88b','#e88','#6b6'])
ax[0].set_ylabel('全年总购电费/元');ax[0].set_title('固定电价 三方案费用对比(2.1-12.31)')
for k,v in zip(range(3),[yf['q3_A'],yf['q3_B'],yf['q3_C']]):ax[0].text(k,v,f'{v/1e4:.0f}万',ha='center',va='bottom')
ax[1].bar(['只0点(A)','保守调整(B)'],[yf['q3_emerg_kwh_A'],yf['q3_emerg_kwh_B']],color=['#88b','#e88'])
ax[1].set_ylabel('紧急购电量/kWh');ax[1].set_title('紧急购电量对比')
for k,v in zip(range(2),[yf['q3_emerg_kwh_A'],yf['q3_emerg_kwh_B']]):ax[1].text(k,v,f'{v/1e4:.1f}万',ha='center',va='bottom')
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig2_compare.png'),dpi=130);plt.close()

# 图3 典型日 3-20
i=q3['dates'].index(dt.date(2025,3,20));h=np.arange(144)/6
fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
ax[0].plot(h,q3['gplan'][i]/DT,label='0点计划购电',lw=1.5)
ax[0].plot(h,q3['gfinal'][i]/DT,label='调整后购电',lw=1.2,alpha=.8)
ax[0].plot(h,q3['emerg'][i]/DT,color='red',label='紧急购电',lw=1)
ax[0].set_ylabel('功率kW');ax[0].legend();ax[0].set_title('2025-03-20 计划/调整/紧急 购电功率')
ax[1].plot(h,q3['P'][i],color='orange',label='实际光伏')
a3=load_a3();fc={hh:fc_to_10min(a3[DATES[0]][hh],hh) for hh in [0,6,12,18]}
for hh,c in [(0,'b'),(6,'g'),(12,'m'),(18,'k')]:ax[1].plot(h,fc[hh],'--',color=c,lw=.9,label=f'{hh}点预报')
ax[1].set_ylabel('光伏kW');ax[1].legend(ncol=5,fontsize=8);ax[1].set_xlim(0,24);ax[1].set_xlabel('时')
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig3_typical.png'),dpi=130);plt.close()

with open(os.path.join(OUT,'results_summary.json'),'w',encoding='utf-8') as f:
    json.dump(S,f,ensure_ascii=False,indent=1,default=str)
print(json.dumps(S['year_fixed'],ensure_ascii=False,indent=1))
print(json.dumps(S['year_volatile'],ensure_ascii=False,indent=1))
print("问题1:",json.dumps(S['q1'],ensure_ascii=False))
print("figs & summary saved to",OUT)

# -*- coding: utf-8 -*-
"""结果写出与汇总工具。"""
import os, shutil
import numpy as np, openpyxl
from common import N, DT, PERM
DESK=r"C:\Users\m4113\Desktop"
OUT=os.path.join(os.path.dirname(__file__),"outputs")
os.makedirs(OUT,exist_ok=True)
R=lambda x: round(float(x),4)

def open_template(name):
    src=os.path.join(DESK,name)
    return openpyxl.load_workbook(src)

def save(wb,name):
    p=os.path.join(OUT,name); wb.save(p); return p

def block4(c,d):
    """c,d: (144,) -> 6个4小时时段的(充电,放电)"""
    out=[]
    for k in range(6):
        sl=slice(k*24,(k+1)*24)
        out.append((R(c[sl].sum()),R(d[sl].sum())))
    return out

def merge_emerg(emerg144):
    """emerg144:(144,) 每段紧急购电量; 把连续非零段合并为 (起时分,止时分,合计)"""
    segs=[];i=0
    def hm(m): return f"{m//60}:{m%10:02d}" if False else f"{m//60}:{m%60:02d}"
    while i<N:
        if emerg144[i]>1e-9:
            j=i;s=0.0
            while j<N and emerg144[j]>1e-9:
                s+=emerg144[j];j+=1
            t0=i*10; t1=j*10
            segs.append((f"{t0//60}:{t0%60:02d}-{t1//60}:{t1%60:02d}", R(s)))
            i=j
        else:i+=1
    return segs

# -*- coding: utf-8 -*-
import numpy as np, datetime as dt, openpyxl
from solve_q2 import run
from solve_q3 import build
from writer import write_q2_like
from generate_q3 import write_result3
from io_utils import merge_emerg

# ---- 问题4-2: 波动电价下的问题2 ----
res=run('volatile','q4-2')
p42=write_q2_like('result4-2.xlsx','result4-2.xlsx',res)
print("saved",p42)

# ---- 问题4-3: 波动电价下的问题3 ----
B=build('volatile')
p43=write_result3("result4-3.xlsx","result4-3.xlsx",B)
wb=openpyxl.load_workbook(p43); print("saved",p43)
for sh in wb.sheetnames: print(" ",sh,wb[sh].max_row,"x",wb[sh].max_column)

i0=B['dates'].index(dt.date(2025,2,1)); sl=slice(i0,None)
print("\n=== 波动电价 2.1-12.31 ===")
print("4-2 全年购电费 %.2f 元"%res['daycost'][sl].sum())
print("4-3 A只0点 %.2f | B保守调整 %.2f | C完美 %.2f 元"%(
    B['totA'][sl].sum(),B['totB'][sl].sum(),B['totC'][sl].sum()))
for tgt in [dt.date(2025,3,20),dt.date(2025,6,21),dt.date(2025,9,23),dt.date(2025,12,21)]:
    i=B['dates'].index(tgt); segs=merge_emerg(B['emerg'][i])
    print(tgt,"4-3 计划购电%.0f 最终%.0f 紧急%.1f(%d段) 总费%.1f"%(
        B['gplan'][i].sum(),B['gfinal'][i].sum(),B['emerg'][i].sum(),len(segs),B['totB'][i]))

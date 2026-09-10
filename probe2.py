# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os
D = r"C:\Users\m4113\Desktop"
def show(f):
    p=os.path.join(D,f); xl=pd.ExcelFile(p)
    print("#"*90); print(f, "sheets=",xl.sheet_names)
    for sh in xl.sheet_names:
        df=xl.parse(sh,header=None)
        print(f"\n--- sheet[{sh}] shape={df.shape}")
        print("col0 head:", [str(x)[:12] for x in df.iloc[:6,0].tolist()])
        print("col0 tail:", [str(x)[:12] for x in df.iloc[-4:,0].tolist()])
        print("row0 (col headers) first8:", [str(x)[:10] for x in df.iloc[0,:8].tolist()],
              " last4:", [str(x)[:10] for x in df.iloc[0,-4:].tolist()])
        print("row1 first6:", [str(x)[:11] for x in df.iloc[1,:6].tolist()])

for f in ["附件2.xlsx","附件3.xlsx","附件4.xlsx"]:
    show(f)

# -*- coding: utf-8 -*-
import pandas as pd, os, numpy as np
D=r"C:\Users\m4113\Desktop"
def hdr(f,sh):
    df=pd.read_excel(os.path.join(D,f),sheet_name=sh,header=None)
    print(f"\n### {f} [{sh}] shape={df.shape}")
    print(" row0:", [f"{i}:{str(v)[:9]}" for i,v in enumerate(df.iloc[0,:].tolist())])
    if df.shape[0]>1:
        print(" row1:", [f"{i}:{str(v)[:9]}" for i,v in enumerate(df.iloc[1,:].tolist())])
    return df
for f in ["result2.xlsx","result3.xlsx","result4-2.xlsx","result4-3.xlsx"]:
    xl=pd.ExcelFile(os.path.join(D,f))
    print("="*80,f,xl.sheet_names)
    for sh in xl.sheet_names:
        df=hdr(f,sh)
        if min(df.shape)<=10 or sh in ("充放电量","紧急购电量"):
            print(df.head(9).to_string(max_colwidth=11))
# attachment3 alignment: check 1-1 0:00 forecast nonzero hours vs actual
a3=pd.read_excel(os.path.join(D,"附件3.xlsx"),header=None)
r=a3.iloc[1,2:].tolist()  # 1-1 0:00 forecast h1..h24
print("\n附件3 1-1 0:00 forecast h1..h24:",[round(x,1) if isinstance(x,(int,float,np.floating)) else x for x in r])
r6=a3.iloc[2,2:].tolist(); print("附件3 1-1 6:00 forecast h1..h24:",[round(x,1) if isinstance(x,(int,float,np.floating)) else x for x in r6])

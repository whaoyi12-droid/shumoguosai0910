# -*- coding: utf-8 -*-
import pandas as pd, os
D=r"C:\Users\m4113\Desktop"
pd.set_option('display.width',250)
# ---- result templates full ----
for f in ["result1.xlsx","result2.xlsx","result3.xlsx","result4-2.xlsx","result4-3.xlsx"]:
    xl=pd.ExcelFile(os.path.join(D,f))
    print("#"*95); print(f,"sheets=",xl.sheet_names)
    for sh in xl.sheet_names:
        df=xl.parse(sh,header=None)
        print(f"\n  ===[{sh}] shape={df.shape}")
        # print up to first 12 rows, all columns compactly
        with pd.option_context('display.max_columns',None):
            print(df.head(12).to_string(max_colwidth=12))
# ---- attachment 3 detail ----
print("#"*95); print("附件3 first 10 rows, all cols")
a3=pd.read_excel(os.path.join(D,"附件3.xlsx"),header=None)
with pd.option_context('display.max_columns',None):
    print(a3.head(10).to_string(max_colwidth=9))
print("forecast issue times per day (col1) unique:", a3.iloc[1:13,1].tolist())

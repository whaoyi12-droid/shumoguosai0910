# -*- coding: utf-8 -*-
import pandas as pd, openpyxl, os
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 50)
D = r"C:\Users\m4113\Desktop"
files = ["附件1.xlsx","附件2.xlsx","附件3.xlsx","附件4.xlsx",
         "result1.xlsx","result2.xlsx","result3.xlsx","result4-2.xlsx","result4-3.xlsx"]
for f in files:
    p = os.path.join(D,f)
    print("="*100)
    print("FILE:", f, " exists:", os.path.exists(p))
    try:
        wb = openpyxl.load_workbook(p, data_only=False)
        print("sheets:", wb.sheetnames)
        xl = pd.ExcelFile(p)
        for sh in xl.sheet_names:
            df = xl.parse(sh, header=None, nrows=8)
            full = xl.parse(sh, header=None)
            print("-"*80)
            print(f"  sheet='{sh}'  shape={full.shape}")
            print("  first 8 rows:")
            print(df.to_string(max_colwidth=14))
    except Exception as e:
        print("ERROR:", repr(e))

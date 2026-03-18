import csv
import os

os.chdir("3des-pipeline/Output")

with open("Final_Report_mastercard_session.csv") as f:
    reader = csv.DictReader(f)
    rows_with_3des = 0
    rows_with_rsa = 0
    total_rows = 0
    for row in reader:
        total_rows += 1
        if row['3DES_KENC'] and row['3DES_KENC'].strip():
            rows_with_3des += 1
        if row['RSA_CRT_P'] and row['RSA_CRT_P'].strip():
            rows_with_rsa += 1
    
    print(f"Total rows: {total_rows}")
    print(f"Rows with 3DES keys: {rows_with_3des}")
    print(f"Rows with RSA keys: {rows_with_rsa}")
    print(f"\n3DES recovery rate: {100*rows_with_3des/total_rows if total_rows > 0 else 0:.1f}%")
    print(f"RSA recovery rate: {100*rows_with_rsa/total_rows if total_rows > 0 else 0:.1f}%")

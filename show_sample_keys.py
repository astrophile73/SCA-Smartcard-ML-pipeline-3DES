import csv
import os

os.chdir("3des-pipeline/Output")

with open("Final_Report_mastercard_session.csv") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < 3:
            print(f"Row {i+1}:")
            print(f"  3DES_KENC: {row['3DES_KENC'][:32] if row['3DES_KENC'] else 'EMPTY'}...")
            print(f"  3DES_KMAC: {row['3DES_KMAC'][:32] if row['3DES_KMAC'] else 'EMPTY'}...")
            print(f"  3DES_KDEK: {row['3DES_KDEK'][:32] if row['3DES_KDEK'] else 'EMPTY'}...")
            print(f"  RSA_CRT_P: {row['RSA_CRT_P'][:32] if row['RSA_CRT_P'] else 'EMPTY'}...")
            print()

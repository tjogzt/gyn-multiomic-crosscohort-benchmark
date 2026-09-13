#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n5：核对层矩阵列名与纯度表的 ID 键"""
import glob, os
import pandas as pd
H = "/tmp/gyn_retyping/hcsv"
B = "/Volumes/tjogzt4T/data"
for f in sorted(glob.glob(f"{H}/ind__*.csv.gz"))[:3] + sorted(glob.glob(f"{H}/dis__*.csv.gz"))[:2]:
    M = pd.read_csv(f, index_col=0, nrows=0)
    print(f"{os.path.basename(f):26s} 列数 {M.shape[1]:5d} | 前5: {list(M.columns[:5])}")

print("\n--- ind meta 的 ID 字段 ---")
x = pd.read_excel(f"{B}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
for c in ["Idx", "Case_id", "Aliquot_ID", "Sample_ID", "Case_excluded"]:
    if c in x.columns:
        print(f"  {c:16s} 前5: {list(x[c].astype(str).head(5))}")

print("\n--- dis CLI 的 ID 字段 ---")
d = pd.read_csv(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
for c in ["idx", "Proteomics_Participant_ID", "Proteomics_Aliquot_ID", "Proteomics_Parent_Sample_IDs"]:
    if c in d.columns:
        print(f"  {c:28s} 前5: {list(d[c].astype(str).head(5))}")

print("\n--- 交集检查 ---")
Mind = pd.read_csv(f"{H}/ind__mRNA.csv.gz", index_col=0, nrows=0)
Mdis = pd.read_csv(f"{H}/dis__mRNA.csv.gz", index_col=0, nrows=0)
for nm, cols, key in [("ind Case_id", list(Mind.columns), x["Case_id"].astype(str)),
                      ("ind Aliquot_ID", list(Mind.columns), x["Aliquot_ID"].astype(str)),
                      ("dis Participant", list(Mdis.columns), d["Proteomics_Participant_ID"].astype(str))]:
    inter = len(set(cols) & set(key))
    print(f"  {nm:16s} 矩阵列 {len(cols)} vs 键 {len(key)} → 交集 {inter}")

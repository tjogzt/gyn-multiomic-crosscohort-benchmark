#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n1：纯度数据可得性普查"""
import glob, os
import numpy as np, pandas as pd

def show(path, label, reader=None):
    print("=" * 88); print(label, "|", path)
    try:
        df = reader() if reader else pd.read_csv(path, sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
    except Exception as e:
        print("  读取失败:", str(e)[:120]); return None
    print(f"  形状 {df.shape}")
    cols = [c for c in df.columns]
    pur = [c for c in cols if any(k in c.lower() for k in ["purit", "absolute", "tumor_cell", "stromal", "immune", "leuki"])]
    print(f"  疑似纯度/组分列: {pur if pur else '无'}")
    if pur:
        for c in pur[:6]:
            s = pd.to_numeric(df[c], errors="coerce")
            print(f"    {c}: 非空 {s.notna().sum()}/{len(df)} 均值 {s.mean():.4f} 范围 [{s.min():.3f},{s.max():.3f}]")
    print(f"  全部列（前 30）: {cols[:30]}")
    return df

B = "/Volumes/tjogzt4T/data"
show(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt", "CPTAC UCEC Discovery")
show(f"{B}/cptac/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_CLI.txt", "CPTAC OV Prospective")
print("=" * 88); print("CPTAC UCEC ind（xlsx）")
try:
    x = pd.read_excel(f"{B}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
    print(f"  形状 {x.shape}")
    pur = [c for c in x.columns if any(k in str(c).lower() for k in ["purit", "absolute", "tumor_cell", "stromal", "immune"])]
    print(f"  疑似纯度列: {pur if pur else '无'}")
    if pur:
        for c in pur[:6]:
            s = pd.to_numeric(x[c], errors="coerce")
            print(f"    {c}: 非空 {s.notna().sum()}/{len(x)} 均值 {s.mean():.4f}")
    print(f"  全部列: {list(x.columns)[:40]}")
except Exception as e:
    print("  失败:", str(e)[:150])

print("\n" + "=" * 88); print("Xena 目录")
for f in glob.glob(f"{B}/xena/*/*"):
    n = os.path.basename(f)
    if any(k in n.lower() for k in ["purit", "absolute", "samplemap", "phenotype", "clinical"]):
        print("  ", n, f"({os.path.getsize(f)//1024} KB)")
print("\n  xena 目录一览:")
for d in sorted(glob.glob(f"{B}/xena/*/")):
    print("   ", d, "->", len(glob.glob(d + "*")), "文件")

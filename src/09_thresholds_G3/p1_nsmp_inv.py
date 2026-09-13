#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p1：NSMP 可得性清点（G3 冻结的前置）"""
import glob, os, json
import numpy as np, pandas as pd

B = "/Volumes/tjogzt4T/data"
T = "/tmp/gyn_retyping"

print("=" * 90); print("① 找既往 NSMP 产物"); print("=" * 90)
pats = [f"{T}/*nsmp*", f"{T}/*NSMP*", f"{T}/**/*nsmp*", f"{T}/**/*NSMP*",
        os.path.expanduser("~/my researches/retyping/*NSMP*"),
        os.path.expanduser("~/my researches/retyping/*nsmp*")]
seen = set()
for p in pats:
    for f in glob.glob(p, recursive=True):
        if f in seen: continue
        seen.add(f)
        print(f"  {f}  ({os.path.getsize(f)} B)")
if not seen: print("  （无既往 NSMP 文件）")

print("\n" + "=" * 90); print("② 各队列的分子分型字段"); print("=" * 90)
# CPTAC ind
x = pd.read_excel(f"{B}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
print(f"CPTAC ind (V1) 形状 {x.shape}")
for c in ["Genomic_subtype", "CNV_status", "MSI_status", "POLE", "Histologic_Type"]:
    if c in x.columns:
        print(f"  {c}: {x[c].value_counts(dropna=False).to_dict()}")
# CPTAC dis
dc = pd.read_csv(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                 sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
print(f"\nCPTAC dis (V2) 形状 {dc.shape}")
cand = [c for c in dc.columns if any(k in c.lower() for k in ["subtype", "cnv", "msi", "pms", "pole", "molecul"])]
print("  候选分型列:", cand)
for c in cand[:6]:
    print(f"  {c}: {dc[c].value_counts(dropna=False).head(6).to_dict()}")

print("\n" + "=" * 90); print("③ TCGA 侧的分型来源"); print("=" * 90)
uc = f"{B}/xena/UCEC/TCGA.UCEC.sampleMap_UCEC_clinicalMatrix"
if os.path.exists(uc):
    hdr = pd.read_csv(uc, sep="\t", nrows=0)
    cols = [c for c in hdr.columns if any(k in c.lower() for k in ["subtype", "cnv", "msi", "integrated", "molecul"])]
    print(f"  clinicalMatrix 列数 {len(hdr.columns)} | 候选: {cols}")
    d = pd.read_csv(uc, sep="\t", low_memory=False, usecols=["sampleID"] + cols if cols else None)
    for c in cols:
        print(f"  {c}: {d[c].value_counts(dropna=False).head(6).to_dict()}")
sf = f"{B}/xena/tcgapancan/TCGASubtype.20170308.tsv.gz"
if os.path.exists(sf):
    d2 = pd.read_csv(sf, sep="\t", low_memory=False)
    print(f"\n  TCGASubtype.20170308: {d2.shape} | 列 {list(d2.columns)[:8]}")
    for c in d2.columns:
        if "subtype" in c.lower():
            print(f"    {c}: {d2[c].value_counts(dropna=False).head(8).to_dict()}")

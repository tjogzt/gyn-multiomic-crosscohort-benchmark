#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l0：修复 protein_gene.csv（k1 中 reindex 用错索引导致基因列全为 NaN）"""
import csv
import numpy as np, pandas as pd

csv.field_size_limit(10**9)

SRC = "/Volumes/tjogzt4T/data/pride/PXD046507/combined__txt__proteinGroups.txt"
E = "/tmp/gyn_retyping/eeec_batch"

Lm = pd.read_csv(f"{E}/matrix_log2.csv", index_col=0)
want = set(Lm.index.astype(str))
print("目标蛋白组:", len(want))

pmap = {}
with open(SRC, newline="", encoding="utf-8", errors="replace") as f:
    rd = csv.reader(f, delimiter="\t")
    hdr = next(rd)
    ip = hdr.index("Protein IDs"); ig = hdr.index("Gene names")
    for row in rd:
        if len(row) <= max(ip, ig): continue
        pid = row[ip]
        if pid in want:
            g = (row[ig] or "").split(";")[0].strip()
            pmap[pid] = g
print("从源文件取到:", len(pmap), "| 有基因名:", sum(1 for v in pmap.values() if v))

old = pd.read_csv(f"{E}/protein_gene.csv")
print("旧文件非空基因数（按真值）:", int((~old['gene'].isna() & (old['gene'].astype(str).str.strip()!='') & (old['gene'].astype(str)!='nan')).sum()))

out = pd.DataFrame({"protein": list(want), "gene": [pmap.get(p, "") for p in want]})
out.to_csv(f"{E}/protein_gene.csv", index=False)
print("已重建 protein_gene.csv | 非空:", int((out['gene'] != '').sum()), "/", len(out))
print(out.head(4).to_string(index=False))
print("\n唯一基因数:", out.loc[out['gene'] != '', 'gene'].nunique())

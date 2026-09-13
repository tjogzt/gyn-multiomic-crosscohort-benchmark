#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p3：TCGA 矩阵列名与 NSMP ID 的对齐"""
import json, os
import pandas as pd
T = "/tmp/gyn_retyping"; H = f"{T}/hcsv"
ns = json.load(open(f"{T}/ucec_core_nsmp.json"))
core, nsmp = set(ns["core"]), set(ns["nsmp_core"])
print(f"core {len(core)} | nsmp_core {len(nsmp)} | nsmp ⊆ core ? {nsmp <= core}")
print("nsmp 样例:", sorted(nsmp)[:4])

for lay in ["mRNA", "miRNA", "CNA", "meth"]:
    p = f"{H}/TCGA__{lay}.csv.gz"
    if not os.path.exists(p): print(f"  {lay}: 缺"); continue
    M = pd.read_csv(p, index_col=0, nrows=0)
    cols = [str(c) for c in M.columns]
    print(f"\n{lay}: 列数 {len(cols)}")
    print("  前3列:", cols[:3])
    print("  后3列:", cols[-3:])
    for tag, S in [("nsmp_core 12字符", {s[:12] for s in nsmp}), ("core 12字符", {s[:12] for s in core})]:
        hit = len({c[:12] for c in cols} & S)
        print(f"  按前12字符匹配 {tag}: 命中 {hit}")

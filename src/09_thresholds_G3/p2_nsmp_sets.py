#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p2：三队列 NSMP 例数与层可用性核验"""
import json, os
import numpy as np, pandas as pd

B = "/Volumes/tjogzt4T/data"
T = "/tmp/gyn_retyping"
H = f"{T}/hcsv"

print("=== 既往 TCGA NSMP 产物 ===")
ns = json.load(open(f"{T}/ucec_core_nsmp.json"))
print("类型:", type(ns).__name__, "| 键/长度:", list(ns.keys())[:6] if isinstance(ns, dict) else len(ns))
if isinstance(ns, dict):
    for k, v in ns.items():
        print(f"  {k}: {type(v).__name__} 长度 {len(v) if hasattr(v,'__len__') else v}")
        if isinstance(v, list) and v: print(f"     样例 {v[:4]}")
    ids = None
    for k in ns:
        if isinstance(ns[k], list) and len(ns[k]) > 30 and all(isinstance(z, str) for z in ns[k][:5]):
            ids = ns[k]; print(f"  → 用作 NSMP 列表: 键 '{k}'，{len(ids)} 例"); break
else:
    ids = ns; print("  样例:", list(ids)[:4])

def L(coh, lay):
    p = f"{H}/{coh}__{lay}.csv.gz"
    if not os.path.exists(p): return None
    M = pd.read_csv(p, index_col=0); M.columns = [str(c) for c in M.columns]
    return M

print("\n=== TCGA NSMP 在各层的可用性 ===")
tc = set(ids) if ids else set()
print(f"  NSMP 名单 {len(tc)} 例")
tot = 0
for lay in ["mRNA", "miRNA", "CNA", "meth"]:
    M = L("TCGA", lay)
    if M is None: print(f"  {lay}: 层缺失"); continue
    have = tc & set(M.columns)
    print(f"  {lay:7s}: 全层 {M.shape[1]:4d} | NSMP 命中 {len(have):3d}")
# 四层交集
inter = tc.copy()
for lay in ["mRNA", "miRNA", "CNA", "meth"]:
    M = L("TCGA", lay)
    if M is not None: inter &= set(M.columns)
print(f"  ** 四层交集 NSMP: {len(inter)} 例")

print("\n=== V1 (CPTAC ind) NSMP ===")
x = pd.read_excel(f"{B}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
x["id"] = x["Case_id"].astype(str)
v1_ids = set(x.loc[x["Genomic_subtype"] == "CNV_L", "id"])
print(f"  Genomic_subtype == CNV_L: {len(v1_ids)} 例")
i1 = v1_ids.copy()
for lay in ["mRNA", "miRNA", "CNA", "meth"]:
    M = L("ind", lay)
    if M is None: print(f"  {lay}: 层缺失"); continue
    h = v1_ids & set(M.columns)
    print(f"  {lay:7s}: NSMP 命中 {len(h):3d}")
    i1 &= set(M.columns)
print(f"  ** 四层交集: {len(i1)} 例")

print("\n=== V2 (CPTAC dis) NSMP ===")
dc = pd.read_csv(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                 sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
dc["id"] = dc["Proteomics_Participant_ID"].astype(str)
v2_ids = set(dc.loc[dc["CNV_class"] == "CNV_LOW", "id"])
print(f"  CNV_class == CNV_LOW: {len(v2_ids)} 例")
i2 = v2_ids.copy()
for lay in ["mRNA", "miRNA", "CNA", "meth"]:
    M = L("dis", lay)
    if M is None: print(f"  {lay}: 层缺失"); continue
    h = v2_ids & set(M.columns)
    print(f"  {lay:7s}: NSMP 命中 {len(h):3d}")
    i2 &= set(M.columns)
print(f"  ** 四层交集: {len(i2)} 例")

out = {"TCGA_nsmp_all": sorted(tc), "TCGA_nsmp_4layer": sorted(inter),
       "V1_nsmp_4layer": sorted(i1), "V2_nsmp_4layer": sorted(i2),
       "counts": {"TCGA_list": len(tc), "TCGA_4L": len(inter), "V1_4L": len(i1), "V2_4L": len(i2)}}
json.dump(out, open(f"{T}/nsmp_sets.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 nsmp_sets.json")
print("计数:", out["counts"])

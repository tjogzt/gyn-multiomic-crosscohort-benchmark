#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""k1：EEEC 批次测试床——构建 2x2 平衡分析矩阵（批次 E/L × 条件 can/norm，各 49 例）"""
import csv, os, json
import numpy as np, pandas as pd

SRC = "/Volumes/tjogzt4T/data/pride/PXD046507/combined__txt__proteinGroups.txt"
OUT = "/tmp/gyn_retyping/eeec_batch"
os.makedirs(OUT, exist_ok=True)

with open(SRC, newline="", encoding="utf-8", errors="replace") as f:
    hdr = next(csv.reader(f, delimiter="\t"))
lfq = [c for c in hdr if c.startswith("LFQ intensity ")]
keep = ["Protein IDs", "Gene names", "Reverse", "Potential contaminant", "Only identified by site"]

print("读取目标列 …")
D = pd.read_csv(SRC, sep="\t", usecols=keep + lfq, low_memory=False,
                encoding="utf-8", encoding_errors="replace")
print("原始行数:", len(D))

flt = ((D["Reverse"] != "+") & (D["Potential contaminant"] != "+") &
       (D["Only identified by site"] != "+"))
D = D[flt].copy()
print("过滤 reverse/contaminant/only-site 后:", len(D))

X = D[lfq].apply(pd.to_numeric, errors="coerce")
X[X <= 0] = np.nan                      # LFQ 0 = 缺失
X.index = D["Protein IDs"].astype(str)
genes = D["Gene names"].astype(str).str.split(";").str[0].str.strip()
gene_counts = genes.value_counts()
print(f"蛋白组 {X.shape[0]} × 样本 {X.shape[1]} | 有基因名 {int((genes!='').sum())}")

# 样本 → (批次, 条件)
def parse(c):
    b = c.replace("LFQ intensity ", "").strip()
    for fam in ["Ecan", "Enorm", "Lcan", "Lnorm", "Llymp", "baoyu"]:
        if b.startswith(fam):
            return fam, b[len(fam):]
    return "other", b
meta = pd.DataFrame([{"col": c, **dict(zip(["family", "suffix"], parse(c)))} for c in lfq])
meta["batch"] = meta["family"].str[0]
meta["cond"] = meta["family"].str[1:]
print("\n族 × 批次 × 条件:")
print(meta.groupby(["family", "batch", "cond"]).size().to_string())

# 主分析子集：4 个平衡族
SEL = ["Ecan", "Enorm", "Lcan", "Lnorm"]
m = meta[meta["family"].isin(SEL)].copy()
print(f"\n主分析样本数: {len(m)}  ({m.groupby(['batch','cond']).size().to_dict()})")

Xm = X[m["col"].tolist()]
Xm = Xm.loc[:, m["col"].tolist()]

# 完整度筛选：在每个 (batch × cond) 单元内 ≥80% 非缺失
ok = pd.Series(True, index=Xm.index)
for (b, c), g in m.groupby(["batch", "cond"]):
    sub = Xm[g["col"].tolist()]
    ok &= (sub.notna().mean(axis=1) >= 0.8)
Xm = Xm[ok]
print(f"每单元 ≥80% 完整度筛选后: {Xm.shape[0]} 蛋白组")
print(f"缺失率: {Xm.isna().mean().mean()*100:.2f}%")

# 剩余缺失用该 (蛋白 × 单元) 中位数填补（仅 2.9% 量级）
for (b, c), g in m.groupby(["batch", "cond"]):
    cols = g["col"].tolist()
    med = Xm[cols].median(axis=1)
    Xm[cols] = Xm[cols].apply(lambda r: r.fillna(med), axis=0)
print(f"填补后缺失: {int(Xm.isna().sum().sum())}")

L = np.log2(Xm)
L.index.name = "protein"
L.to_csv(f"{OUT}/matrix_log2.csv")
m.to_csv(f"{OUT}/sample_meta.csv", index=False)
pd.DataFrame({"protein": Xm.index, "gene": genes.reindex(Xm.index).values}).to_csv(f"{OUT}/protein_gene.csv", index=False)

info = {"n_protein": int(L.shape[0]), "n_sample": int(L.shape[1]),
        "design": {f"{b}_{c}": int(n) for (b, c), n in m.groupby(["batch", "cond"]).size().items()},
        "missing_pct_after_filter": float(Xm.isna().mean().mean()*100)}
json.dump(info, open(f"{OUT}/info.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘:", OUT)
print(json.dumps(info, ensure_ascii=False, indent=1))

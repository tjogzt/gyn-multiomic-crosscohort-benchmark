#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEEC-19：蛋白组样本语义验证 —— can/norm 是否对应 肿瘤/正常"""
import re, numpy as np, pandas as pd
from collections import Counter
P = "/Volumes/tjogzt4T/data/pride/PXD046507/combined__txt__proteinGroups.txt"
print("读取 proteinGroups（仅取关键列）…")
with open(P, encoding="utf-8", errors="replace") as f:
    hdr = f.readline().rstrip("\n").split("\t")
idx = {c: i for i, c in enumerate(hdr)}
L = [i for i, c in enumerate(hdr) if c.startswith("LFQ intensity ")]
names = [hdr[i].replace("LFQ intensity ", "") for i in L]
gi, ri, ci = idx["Gene names"], idx["Reverse"], idx["Potential contaminant"]
us, up = idx["Unique peptides"], idx.get("Only identified by site")
rows, genes = [], []
with open(P, encoding="utf-8", errors="replace") as f:
    f.readline()
    for line in f:
        c = line.rstrip("\n").split("\t")
        if len(c) <= max(L): continue
        if c[ri].strip() == "+" or c[ci].strip() == "+": continue
        if up is not None and c[up].strip() == "+": continue
        try:
            if float(c[us] or 0) < 2: continue
        except Exception: continue
        g = c[gi].strip().split(";")[0].strip()
        if not g: continue
        rows.append([c[i] for i in L]); genes.append(g)
print(f"净蛋白组（≥2 unique peptides）: {len(rows)}")
M = pd.DataFrame(np.array(rows, dtype=float), index=genes, columns=names)
M = M.groupby(level=0).mean()
X = np.log2(M.replace(0, np.nan))
X = X.dropna(axis=1, how="all")
X = X.sub(X.median(axis=0), axis=1)          # 逐样本中位归一
X = X.dropna(axis=0, thresh=int(0.6 * X.shape[1]))
print(f"矩阵: {X.shape[0]} 蛋白 × {X.shape[1]} 样本；缺失率 {100*(1-X.notna().mean().mean()):.1f}%")

fam = lambda n: re.match(r"([A-Za-z]+)", n).group(1)
groups = {}
for f in ["Ecan", "Enorm", "Lcan", "Lnorm", "Llymp", "baoyu"]:
    cs = [c for c in X.columns if fam(c) == f]
    if cs: groups[f] = cs
print("\n▍家族样本数:", {k: len(v) for k, v in groups.items()})

# 组均值谱 + 相关
gm = {k: X[v].mean(axis=1) for k, v in groups.items()}
print("\n" + "=" * 96); print("【1】组均值谱两两 Pearson 相关（是否支持 can=肿瘤 / norm=正常）"); print("=" * 96)
keys = list(gm)
print(f"{'':<9}" + "".join(f"{k:>10}" for k in keys))
for a in keys:
    line = f"{a:<9}"
    for b in keys:
        c = gm[a].index.intersection(gm[b].index)
        x, y = gm[a].reindex(c).values, gm[b].reindex(c).values
        m = np.isfinite(x) & np.isfinite(y)
        line += f"{np.corrcoef(x[m], y[m])[0,1]:>10.4f}" if m.sum() > 100 else f"{'—':>10}"
    print(line)

print("\n" + "=" * 96); print("【2】整体聚类结构（样本前 2 主成分的组分布）"); print("=" * 96)
Xf = X.sub(X.mean(axis=1), axis=0).fillna(0)
u, s, vt = np.linalg.svd(Xf.sub(Xf.mean(axis=1), axis=0).values, full_matrices=False)
pcs = u[:, :2] * s[:2]
print(f"  PC1 解释方差 {s[0]**2/np.sum(s**2)*100:.1f}% | PC2 {s[1]**2/np.sum(s**2)*100:.1f}%")
for k, cs in groups.items():
    ii = [Xf.columns.get_loc(c) for c in cs]
    print(f"  {k:<9} PC1 均值 {pcs[ii,0].mean():+8.3f}  PC2 均值 {pcs[ii,1].mean():+8.3f}   n={len(cs)}")

print("\n" + "=" * 96); print("【3】can vs norm 的组间差异检验（应显著）"); print("=" * 96)
try:
    from scipy.stats import mannwhitneyu
    for a, b in [("Ecan", "Enorm"), ("Lcan", "Lnorm")]:
        if a in groups and b in groups:
            d = []
            for g in X.index:
                xa = X.loc[g, groups[a]].dropna().values
                xb = X.loc[g, groups[b]].dropna().values
                if len(xa) >= 5 and len(xb) >= 5:
                    d.append(mannwhitneyu(xa, xb).pvalue)
            d = np.array(d); ok = np.isfinite(d)
            print(f"  {a} vs {b}: 检验蛋白 {ok.sum()} | p<0.05 比例 {100*np.mean(d[ok]<0.05):.1f}% | 中位 p {np.median(d[ok]):.2e}")
except Exception as e:
    print("  检验失败:", str(e)[:120])

print("\n" + "=" * 96); print("【4】样本间相关：同家族内 vs 跨家族"); print("=" * 96)
C = X.sub(X.mean(axis=1), axis=0).fillna(0).corr(method="spearman")
import itertools
for a, b in [("Ecan", "Ecan"), ("Enorm", "Enorm"), ("Ecan", "Enorm"), ("Ecan", "Lcan"), ("baoyu", "baoyu"), ("baoyu", "Ecan")]:
    if a in groups and b in groups:
        A, B = groups[a], groups[b]
        vals = []
        for i, ca in enumerate(A):
            for cb in B:
                if a == b and A.index(ca) >= A.index(cb): continue
                vals.append(C.loc[ca, cb])
        if vals:
            print(f"  {a:<8}×{b:<8} 中位 Spearman {np.median(vals):.4f}   n对 {len(vals)}")
X.to_pickle("/tmp/gyn_retyping/eeec_proteome_log2.pkl")
print("\n已落盘 eeec_proteome_log2.pkl")

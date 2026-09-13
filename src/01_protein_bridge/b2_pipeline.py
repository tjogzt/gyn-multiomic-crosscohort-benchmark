#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""① 三队列 Δ(T−N) 诊断 ② 标准化管线复测"""
import os, gzip, re, json
import numpy as np, pandas as pd, openpyxl
from scipy.stats import spearmanr
np.random.seed(49)
CP = "/Volumes/tjogzt4T/data/cptac"

def load(p):
    op = gzip.open if p.endswith(".gz") else open
    df = pd.read_csv(p, sep="\t", comment="#", header=0, dtype=str, engine="python")
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    df = df[df["ID"].str.match(r"^[A-Za-z][A-Za-z0-9\-\._@]*$", na=False)]
    M = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return M.groupby(level=0).mean()

# ---------- ind 队列按 Group 拆分 ----------
ind = load(f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct")
wb = openpyxl.load_workbook(f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx", read_only=True)
ws = wb[wb.sheetnames[0]]; rows = list(ws.iter_rows(values_only=True)); hdr = [str(x) for x in rows[0]]
gi = hdr.index("Group")
grp = {}
for r in rows[1:]:
    cid = str(r[hdr.index("Case_id")]).strip() if r[hdr.index("Case_id")] else None
    if cid: grp[cid.upper()] = str(r[gi])
colmap = {}
for c in ind.columns:
    k = re.sub(r"-[A-Z]$", "", str(c).upper())
    if k in grp: colmap[c] = grp[k]
Tcols = [c for c, g in colmap.items() if g == "Tumor"]
Ncols = [c for c, g in colmap.items() if g in ("Adjacent_normal", "Enriched_Normal")]
print("ind: 可映射列=%d  肿瘤=%d  正常=%d" % (len(colmap), len(Tcols), len(Ncols)))
indT, indN = ind[Tcols].mean(axis=1, skipna=True), ind[Ncols].mean(axis=1, skipna=True)

disT = load(f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct").mean(axis=1, skipna=True)
disN = load(f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Normal.cct").mean(axis=1, skipna=True)
ovT  = load(f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct").mean(axis=1, skipna=True)
ovN  = load(f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_normal.cct").mean(axis=1, skipna=True)

def sp(a, b, lab):
    c = a.index.intersection(b.index)
    x, y = a.reindex(c).values.astype(float), b.reindex(c).values.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100: print("   %-26s 交集不足(%d)" % (lab, m.sum())); return None
    rho, pv = spearmanr(x[m], y[m])
    print("   %-26s n=%-6d rho=%6.3f  p=%.1e" % (lab, m.sum(), rho, pv))
    return {"n": int(m.sum()), "rho": float(rho), "p": float(pv)}

print("\n" + "=" * 92); print("【A】原始肿瘤基因均值谱"); print("=" * 92)
raw = {}
for lab, a, b in [("dis × ov", disT, ovT), ("dis × ind", disT, indT), ("ov × ind", ovT, indT)]:
    raw[lab] = sp(a, b, lab)

print("\n" + "=" * 92); print("【B】队列内 Δ(T−N)"); print("=" * 92)
Dd, Do, Di = disT - disN, ovT - ovN, indT - indN
dl = {}
for lab, a, b in [("Δdis × Δov", Dd, Do), ("Δdis × Δind", Dd, Di), ("Δov × Δind", Do, Di)]:
    dl[lab] = sp(a, b, lab)

print("\n" + "=" * 92); print("【C】改进倍数"); print("=" * 92)
for k in ["dis × ov", "dis × ind", "ov × ind"]:
    k2 = "Δ" + k.split(" × ")[0].replace("dis", "dis").replace("ov", "ov").replace("ind", "ind") + " × Δ" + k.split(" × ")[1]
    r0 = raw.get(k, {}); r1 = dl.get(k2)
    if r0 and r1:
        print("   %-14s 原始 %.3f → Δ(T−N) %.3f   (×%.1f)" % (k, r0["rho"], r1["rho"], r1["rho"]/max(r0["rho"], 1e-6)))
json.dump({"raw": raw, "delta": dl}, open("/tmp/gyn_retyping/bridge_3cohort.json", "w"), indent=1, ensure_ascii=False)

# ---------- 标准化管线复测 ----------
print("\n" + "=" * 92); print("【D】标准化策略对跨队列一致性的影响（表达层，4 队列）"); print("=" * 92)
TX = "/Volumes/tjogzt4T/data/xena"
EX = {
 "TCGA": f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
 "ind":  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
 "dis":  f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
 "ov":   f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
}
EM = {k: load(p) for k, p in EX.items()}
for k, v in EM.items(): print("   %-6s %d 基因 × %d 样本" % (k, v.shape[0], v.shape[1]))

def qnorm(M):
    """样本级分位数标准化"""
    A = M.values.astype(float).copy()
    nz = np.isfinite(A)
    order = np.argsort(np.where(nz, A, np.nan), axis=0)
    ranks = np.empty_like(A, dtype=float); ranks[:] = np.nan
    for j in range(A.shape[1]):
        col = A[:, j]; ok = np.isfinite(col)
        if ok.sum() == 0: continue
        v = np.sort(col[ok]); r = np.empty(ok.sum()); r[np.argsort(col[ok])] = v
        ranks[ok, j] = r
    return pd.DataFrame(ranks, index=M.index, columns=M.columns)

def ranktrans(M):
    A = M.values.astype(float)
    R = np.full_like(A, np.nan)
    for j in range(A.shape[1]):
        col = A[:, j]; ok = np.isfinite(col)
        if ok.sum() < 2: continue
        R[ok, j] = pd.Series(col[ok]).rank().values
    return pd.DataFrame(R, index=M.index, columns=M.columns)

for strat, fn in [("原始", lambda M: M), ("分位数标准化(qnorm)", qnorm), ("逐样本秩(rank)", ranktrans)]:
    P = {k: fn(v).mean(axis=1, skipna=True) for k, v in EM.items()}
    print("\n   【%s】" % strat)
    for a, b in [("TCGA", "ind"), ("TCGA", "ov"), ("ind", "ov"), ("ind", "dis")]:
        c = P[a].index.intersection(P[b].index)
        x, y = P[a].reindex(c).values, P[b].reindex(c).values
        m = np.isfinite(x) & np.isfinite(y)
        rho, _ = spearmanr(x[m], y[m])
        print("      %-12s n=%-6d rho=%.3f" % ("%s × %s" % (a, b), m.sum(), rho))

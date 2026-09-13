#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""① 队列内 mRNA–蛋白 一致性锚 ② TCGA 450K 探针→基因聚合并与 CPTAC 甲基化对比"""
import os, gzip, re, json
import numpy as np, pandas as pd, openpyxl
from scipy.stats import spearmanr
np.random.seed(49)
CP = "/Volumes/tjogzt4T/data/cptac"; TX = "/Volumes/tjogzt4T/data/xena"

def load(p, cols=None):
    op = gzip.open if p.endswith(".gz") else open
    df = pd.read_csv(p, sep="\t", comment="#", header=0, dtype=str, engine="python", usecols=cols)
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    df = df[df["ID"].str.match(r"^[A-Za-z][A-Za-z0-9\-\._@]*$", na=False)]
    M = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return M.groupby(level=0).mean()

# ---- ind 的 Group 拆分 ----
wb = openpyxl.load_workbook(f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx", read_only=True)
ws = wb[wb.sheetnames[0]]; rows = list(ws.iter_rows(values_only=True)); hdr = [str(x) for x in rows[0]]
gi, ci = hdr.index("Group"), hdr.index("Case_id")
grp = {str(r[ci]).strip().upper(): str(r[gi]) for r in rows[1:] if r[ci]}
def split_by_group(M):
    mp = {}
    for c in M.columns:
        k = re.sub(r"-[A-Z]$", "", str(c).upper())
        if k in grp: mp[c] = grp[k]
    T = [c for c, g in mp.items() if g == "Tumor"]
    N = [c for c, g in mp.items() if g in ("Adjacent_normal", "Enriched_Normal")]
    return M[T].mean(axis=1, skipna=True), M[N].mean(axis=1, skipna=True)

def sp(a, b, lab):
    c = a.index.intersection(b.index)
    x, y = a.reindex(c).values.astype(float), b.reindex(c).values.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100: return None
    rho, pv = spearmanr(x[m], y[m])
    print("   %-30s n=%-6d rho=%6.3f  p=%.1e" % (lab, m.sum(), rho, pv))
    return {"n": int(m.sum()), "rho": float(rho), "p": float(pv)}

print("=" * 96); print("【E】队列内 mRNA–蛋白 一致性（Δ(T−N) 层面，生物学锚）"); print("=" * 96)
out = {}
# dis
dM = load(f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct").mean(axis=1, skipna=True)
dN = load(f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Normal.cct").mean(axis=1, skipna=True)
dR = load(f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct").mean(axis=1, skipna=True)
dRn= load(f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Normal.cct").mean(axis=1, skipna=True)
out["dis_dProt_dRNA"] = sp(dM - dN, dR - dRn, "dis: Δ蛋白 × ΔmRNA")
out["dis_raw_Prot_RNA"] = sp(dM, dR, "dis: 原始蛋白 × 原始mRNA")
# ov
oM = load(f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct").mean(axis=1, skipna=True)
oN = load(f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_normal.cct").mean(axis=1, skipna=True)
oR = load(f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_rnaseq_fpkm_log2.cct").mean(axis=1, skipna=True)
out["ov_raw_Prot_RNA"] = sp(oM, oR, "ov: 原始蛋白 × 原始mRNA（无正常RNA）")
# ind
iM = load(f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct")
iR = load(f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct")
iMT, iMN = split_by_group(iM); iRT, iRN = split_by_group(iR)
out["ind_dProt_dRNA"] = sp(iMT - iMN, iRT - iRN, "ind: Δ蛋白 × ΔmRNA")
out["ind_raw_Prot_RNA"] = sp(iMT, iRT, "ind: 原始蛋白 × 原始mRNA")

print("\n" + "=" * 96); print("【F】TCGA 450K 探针→基因聚合"); print("=" * 96)
am = f"{TX}/tcgapancan/illuminaMethyl450_hg19_GPL16304_TCGAlegacy"
p2g = {}
with open(am, errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1].strip() not in ("", "."):
            p2g[p[0].strip()] = [x for x in p[1].split(",") if x]
print("   映射表: %d 探针 -> %d 基因" % (len(p2g), len({g for v in p2g.values() for g in v})))

p450 = f"{TX}/UCEC/TCGA.UCEC.sampleMap_HumanMethylation450.gz"
acc = []; nread = 0; nkeep = 0
for ch in pd.read_csv(p450, sep="\t", comment="#", header=0, dtype=str, engine="python", chunksize=40000):
    nread += len(ch)
    ids = ch[ch.columns[0]].astype(str).str.strip()
    keep = ch[ids.isin(p2g.keys())]
    nkeep += len(keep)
    if len(keep): acc.append(keep)
print("   扫描行=%d  命中映射表=%d (%.1f%%)" % (nread, nkeep, 100*nkeep/max(nread,1)))
A = pd.concat(acc); A = A.rename(columns={A.columns[0]: "probe"})
num = A.set_index("probe").apply(pd.to_numeric, errors="coerce")
# 探针 -> 基因（多基因探针按各基因分摊，简化为取第一个基因）
g2v = {}
for probe, row in num.iterrows():
    for g in p2g.get(probe, [])[:1]:
        g2v.setdefault(g, []).append(row.values)
GM = pd.DataFrame({g: np.nanmean(np.vstack(v), axis=0) for g, v in g2v.items()}).T
GM.columns = num.columns
print("   TCGA-UCEC 基因级甲基化: %d 基因 × %d 样本" % GM.shape)
gp = GM.mean(axis=1, skipna=True)
print("   基因均值 范围[%.3f, %.3f] 中位=%.3f" % (gp.min(), gp.max(), gp.median()))
GM.to_csv("/tmp/gyn_retyping/tcga450k_genelevel_ucec.csv.gz", compression="gzip")

print("\n" + "=" * 96); print("【G】甲基化跨队列一致性（含 TCGA）"); print("=" * 96)
mI = load(f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct").mean(axis=1, skipna=True)
mD = load(f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct").mean(axis=1, skipna=True)
out["meth_TCGA_ind"] = sp(gp, mI, "TCGA450K(基因级) × CPTAC-ind")
out["meth_TCGA_dis"] = sp(gp, mD, "TCGA450K(基因级) × CPTAC-dis")
out["meth_ind_dis"]   = sp(mI, mD, "CPTAC-ind × CPTAC-dis")
json.dump(out, open("/tmp/gyn_retyping/b3_anchors.json", "w"), indent=1, ensure_ascii=False)
print("\n已落盘 b3_anchors.json + tcga450k_genelevel_ucec.csv.gz")

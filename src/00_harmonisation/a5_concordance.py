#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段4：跨队列同层基因均值谱一致性（Spearman）→ 整合可行性定量判定"""
import os, re, gzip, json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
np.random.seed(49)
CP = "/Volumes/tjogzt4T/data/cptac"; TX = "/Volumes/tjogzt4T/data/xena"

def load(p, max_rows=None, max_cols=250):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            cols = line.rstrip("\n").split("\t"); break
    nc = len(cols) - 1
    use = [0] + (sorted(np.random.choice(range(1, nc+1), max_cols, replace=False)) if nc > max_cols else list(range(1, nc+1)))
    df = pd.read_csv(p, sep="\t", comment="#", header=0, usecols=use, nrows=max_rows,
                     dtype=str, engine="python")
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    df = df[(df["ID"] != "") & df["ID"].str.match(r"^[A-Za-z][A-Za-z0-9\-\._@]*$")]
    M = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    ndup = int(M.index.duplicated().sum())
    if ndup: M = M.groupby(level=0).mean()
    globals().setdefault("DUPS", {})[os.path.basename(p)[:40]] = ndup
    return M.mean(axis=1, skipna=True)

LAYERS = {
 "表达": {
   "TCGA-UCEC":  f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
   "CPTAC-ind":  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
   "CPTAC-dis":  f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
   "CPTAC-OV":   f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
 },
 "蛋白": {
   "CPTAC-ind":  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
   "CPTAC-dis":  f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
   "CPTAC-OV":   f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct",
 },
 "拷贝数": {
   "TCGA-UCEC":  f"{TX}/UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
   "CPTAC-ind":  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
   "CPTAC-dis":  f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
   "CPTAC-OV":   f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_cnv_gene.cct",
 },
 "甲基化": {
   "CPTAC-ind":  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
   "CPTAC-dis":  f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
 },
}
MEANS = {}
print("=" * 104)
print("跨队列同层：基因均值谱 Spearman（队列间无共享样本，故在基因轴上比）")
print("=" * 104)
out = {}
for lay, fs in LAYERS.items():
    print("\n【%s】" % lay)
    ser = {}
    for coh, p in fs.items():
        if not os.path.exists(p): print("   %-12s 文件缺失" % coh); continue
        s = load(p)
        ser[coh] = s
        print("   %-12s 基因数=%d  均值范围[%.2f, %.2f]  中位=%.2f" % (
            coh, len(s), float(np.nanmin(s)), float(np.nanmax(s)), float(np.nanmedian(s))))
    MEANS[lay] = {k: v.to_dict() for k, v in ser.items()}
    ks = list(ser)
    for a, b in itertools.combinations(ks, 2):
        common = ser[a].index.intersection(ser[b].index)
        if len(common) < 100: print("   %-12s × %-12s 交集仅 %d，跳过" % (a, b, len(common))); continue
        x = ser[a].reindex(common).values.astype(float)
        y = ser[b].reindex(common).values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        rho, pv = spearmanr(x[m], y[m])
        out.setdefault(lay, {})["%s|%s" % (a, b)] = {"n": int(m.sum()), "rho": float(rho), "p": float(pv)}
        print("   %-12s × %-12s n=%-6d rho=%6.3f  p=%.1e  %s" % (
            a, b, m.sum(), rho, pv, "✅高" if rho > 0.8 else ("◐中" if rho > 0.5 else "❌低")))
json.dump(out, open("/tmp/gyn_retyping/align_concordance.json", "w"), indent=1, ensure_ascii=False)
print("\n已落盘 align_concordance.json")

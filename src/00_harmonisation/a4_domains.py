#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段3：数值域刻画 + 队列内样本层覆盖 + 跨队列分布可比性"""
import os, re, gzip, json
import numpy as np, pandas as pd
np.random.seed(49)
CP = "/Volumes/tjogzt4T/data/cptac"; TX = "/Volumes/tjogzt4T/data/xena"

def load_sampled(p, max_rows=4000, max_cols=None):
    op = gzip.open if p.endswith(".gz") else open
    # 读表头
    with op(p, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            cols = line.rstrip("\n").split("\t"); break
    ncol = len(cols) - 1
    usecols = list(range(ncol + 1))
    if max_cols and ncol > max_cols:
        usecols = [0] + sorted(np.random.choice(range(1, ncol + 1), max_cols, replace=False))
    df = pd.read_csv(p, sep="\t", comment="#", header=0, usecols=usecols,
                     nrows=max_rows, dtype=str, engine="python")
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    df = df[df["ID"] != ""]
    M = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return M, ncol

FILES = {
 ("TCGA-UCEC","mRNA"):        f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
 ("TCGA-UCEC","甲基化450K"):   f"{TX}/UCEC/TCGA.UCEC.sampleMap_HumanMethylation450.gz",
 ("TCGA-UCEC","RPPA"):        f"{TX}/UCEC/TCGA.UCEC.sampleMap_RPPA.gz",
 ("TCGA-UCEC","CNA(GISTIC2)"):f"{TX}/UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
 ("CPTAC-UCEC-ind","RNAseq"): f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Proteome"):f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Phospho"):f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_phospho_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Acetyl"): f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_acetyl_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Meth"):   f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
 ("CPTAC-UCEC-ind","CNA(log2)"):f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
 ("CPTAC-UCEC-ind","CNA(GISTIC)"):f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_WGS_cnv_gistic_thresholded_tumor_v3.0.cct",
 ("CPTAC-UCEC-ind","miRNA"):  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_miRNAseq_miRNA_TPM_log2(x+1)_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-dis","RNAseq"): f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
 ("CPTAC-UCEC-dis","Proteome"):f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
 ("CPTAC-UCEC-dis","Meth"):   f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
 ("CPTAC-UCEC-dis","SCNV"):   f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
 ("CPTAC-OV-pro","RNAseq"):   f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
 ("CPTAC-OV-pro","Proteome"): f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct",
 ("CPTAC-OV-pro","CNA"):      f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_cnv_gene.cct",
}
rows = []
print("%-16s %-13s %8s %8s %8s %8s %8s %8s %8s %7s %7s %7s" % (
    "队列","层","样本(列)","非NA%","零%","Q05","Q50","Q95","Max","NaN行","域","占位"))
print("-" * 145)
for (coh, lay), p in FILES.items():
    if not os.path.exists(p):
        print("%-16s %-13s 文件缺失" % (coh, lay)); continue
    try:
        M, ncol = load_sampled(p, max_rows=2500, max_cols=200)
    except Exception as e:
        print("%-16s %-13s 读取失败 %s" % (coh, lay, str(e)[:50])); continue
    V = M.values.astype(float)
    nonna = np.isfinite(V)
    allnz = V[nonna]
    nan_rows = int(np.sum(~nonna.any(axis=1)))
    q = np.nanpercentile(allnz, [5, 50, 95]) if allnz.size else [np.nan]*3
    zero = float(np.mean(allnz == 0)) if allnz.size else np.nan
    # 域推断
    if np.nanmin(allnz) >= 0 and np.nanmax(allnz) <= 1.0 and np.nanmax(allnz) > 0.3: dom = "beta[0,1]"
    elif np.nanmin(allnz) >= -1.01 and np.nanmax(allnz) <= 1.01 and np.nanmin(allnz) < -0.5: dom = "相关/logratio[-1,1]"
    elif np.nanmin(allnz) >= -3 and np.nanmax(allnz) <= 3: dom = "GISTIC[-2,2]"
    elif np.nanmin(allnz) >= 0 and np.nanmax(allnz) >= 10: dom = "log2(count+)"
    elif np.nanmin(allnz) < 0 and np.nanmax(allnz) > 3: dom = "log2ratio"
    else: dom = "其它"
    print("%-16s %-13s %8d %7.1f%% %6.1f%% %8.2f %8.2f %8.2f %8.2f %7d %7s" % (
        coh, lay, ncol, 100*np.mean(nonna), 100*zero,
        q[0], q[1], q[2], np.nanmax(allnz), nan_rows, dom))
    rows.append({"cohort": coh, "layer": lay, "n_col": ncol, "pct_finite": 100*float(np.mean(nonna)),
                 "pct_zero": 100*zero, "q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2]),
                 "min": float(np.nanmin(allnz)), "max": float(np.nanmax(allnz)), "domain": dom})
json.dump(rows, open("/tmp/gyn_retyping/align_domains.json", "w"), indent=1, ensure_ascii=False)
print("\n已落盘 align_domains.json")

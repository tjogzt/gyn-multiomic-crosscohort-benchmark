#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨队列对齐测试 · 阶段1：结构 / ID 体系 / 维度普查"""
import os, re, gzip, json, collections
import numpy as np, pandas as pd
CP = "/Volumes/tjogzt4T/data/cptac"
TX = "/Volumes/tjogzt4T/data/xena"

def read_head(path, nrow=6):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        rows = []
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            rows.append(line.rstrip("\n").split("\t"))
            if len(rows) >= nrow: break
    return rows

def id_kind(ids):
    ids = [x.split("|")[0].strip() for x in ids if x.strip()]
    if not ids: return "空"
    ensg = sum(1 for x in ids if x.upper().startswith("ENSG"))
    if ensg / len(ids) > 0.6: return "ENSG"
    if all(re.fullmatch(r"[A-Za-z0-9\-\._]+", x) for x in ids) and any(x.isupper() for x in ids):
        return "symbol"
    return "混合/其它"

FILES = {
 # 队列, 层名, 路径
 ("TCGA-UCEC","mRNA(EB++Adjust)"):      f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
 ("TCGA-UCEC","甲基化450K"):             f"{TX}/UCEC/TCGA.UCEC.sampleMap_HumanMethylation450.gz",
 ("TCGA-UCEC","RPPA"):                  f"{TX}/UCEC/TCGA.UCEC.sampleMap_RPPA.gz",
 ("TCGA-UCEC","miRNA"):                 f"{TX}/UCEC/TCGA.UCEC.sampleMap_miRNA_HiSeq_gene.gz",
 ("TCGA-UCEC","CNA(GISTIC2)"):          f"{TX}/UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
 ("CPTAC-UCEC-ind","RNAseq(gene)"):     f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Proteome"):         f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Phospho(gene)"):    f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_phospho_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Acetyl(gene)"):     f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_acetyl_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","N-glyco(site)"):    f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_nglycoform-site_ratio_median_centered_log2_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-ind","Methylation"):      f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
 ("CPTAC-UCEC-ind","CNA(GISTIC thr)"):  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_WGS_cnv_gistic_thresholded_tumor_v3.0.cct",
 ("CPTAC-UCEC-ind","CNA(log2)"):        f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
 ("CPTAC-UCEC-ind","miRNA"):            f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_miRNAseq_miRNA_TPM_log2(x+1)_tumor_normal_v3.0.cct",
 ("CPTAC-UCEC-dis","RNAseq(gene)"):     f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
 ("CPTAC-UCEC-dis","Proteome"):         f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
 ("CPTAC-UCEC-dis","Phospho(gene)"):    f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Phosphoproteomics_gene_level_log2_Tumor.cct",
 ("CPTAC-UCEC-dis","Acetyl(gene)"):     f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Acetylproteomics_gene_level_log2_Tumor.cct",
 ("CPTAC-UCEC-dis","Methylation"):      f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
 ("CPTAC-UCEC-dis","SCNV(log2)"):       f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
 ("CPTAC-UCEC-dis","miRNA"):            f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_microRNA_log2_Tumor.cct",
 ("CPTAC-OV-pro","RNAseq(gene)"):       f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
 ("CPTAC-OV-pro","Proteome(tumor)"):    f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct",
 ("CPTAC-OV-pro","Phospho(site,tumor)"):f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_phosphoproteome_site_tumor.cct",
 ("CPTAC-OV-pro","N-glyco(site,tumor)"):f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_unfractionated_nglycopeptide_tumor.cct",
 ("CPTAC-OV-pro","CNA(log2)"):          f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_cnv_gene.cct",
}
rows = []
for (coh, lay), p in FILES.items():
    if not os.path.exists(p):
        rows.append((coh, lay, "缺失", "", "", "", "")); continue
    hd = read_head(p, 4)
    if len(hd) < 2:
        rows.append((coh, lay, "读取失败", "", "", "", "")); continue
    ids = [r[0] for r in hd[1:]]
    kind = id_kind(ids)
    # 全文件行数（流式计数，避免载入）
    op = gzip.open if p.endswith(".gz") else open
    nrow = 0; ncol = len(hd[0]) - 1
    with op(p, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            nrow += 1
    rows.append((coh, lay, kind, nrow - 1, ncol, ids[0][:22], os.path.getsize(p)/1048576))

print("=" * 122)
print("%-16s %-20s %-10s %8s %7s  %-22s %9s" % ("队列","层","ID体系","基因数","样本数","首个ID","MB"))
print("=" * 122)
for r in rows:
    print("%-16s %-20s %-10s %8s %7s  %-22s %9s" % (
        r[0], r[1], r[2], r[3], r[4], r[5], ("%.1f" % r[6]) if isinstance(r[6], float) else r[6]))
json.dump([dict(zip(["cohort","layer","id_kind","genes","samples","first_id","mb"], r)) for r in rows],
          open("/tmp/gyn_retyping/align_stage1.json","w"), indent=1, ensure_ascii=False)
print("\n已落盘 align_stage1.json")

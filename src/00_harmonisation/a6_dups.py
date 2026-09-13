#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重复基因 ID 普查（对齐必须处理的隐藏问题）"""
import os, re, gzip, collections, json
CP = "/Volumes/tjogzt4T/data/cptac"; TX = "/Volumes/tjogzt4T/data/xena"
F = {
 "TCGA mRNA":   f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
 "TCGA CNA":    f"{TX}/UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
 "CPTAC-ind RNAseq":  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
 "CPTAC-ind Proteome":f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 "CPTAC-ind Meth":    f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
 "CPTAC-dis RNAseq":  f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
 "CPTAC-OV RNAseq":   f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
}
res = {}
print("%-22s %8s %8s %8s %10s" % ("文件", "总ID", "唯一", "重复", "重复率"))
for tag, p in F.items():
    if not os.path.exists(p): print("%-22s 缺失" % tag); continue
    op = gzip.open if p.endswith(".gz") else open
    ids = []
    with op(p, "rt", errors="replace") as f:
        first = True
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            if first: first = False; continue
            ids.append(line.split("\t", 1)[0].strip().strip('"'))
    c = collections.Counter(ids)
    dup = sum(v - 1 for v in c.values() if v > 1)
    res[tag] = {"total": len(ids), "uniq": len(c), "dup": dup, "dup_rate": dup/max(len(ids),1)}
    print("%-22s %8d %8d %8d %9.1f%%" % (tag, len(ids), len(c), dup, 100*dup/max(len(ids),1)))
json.dump(res, open("/tmp/gyn_retyping/align_dups.json", "w"), indent=1, ensure_ascii=False)
print("\n已落盘 align_dups.json")

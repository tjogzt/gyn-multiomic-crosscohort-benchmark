#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补查剩余文件重复 ID"""
import os, gzip, collections
CP = "/Volumes/tjogzt4T/data/cptac"; TX = "/Volumes/tjogzt4T/data/xena"
F = {
 "CPTAC-dis Proteome": f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
 "CPTAC-dis Meth":     f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
 "CPTAC-dis SCNV":     f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
 "CPTAC-OV Proteome":  f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct",
 "CPTAC-OV CNA":       f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_cnv_gene.cct",
 "CPTAC-ind CNA(log2)":f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
 "TCGA 450K":          f"{TX}/UCEC/TCGA.UCEC.sampleMap_HumanMethylation450.gz",
}
print("%-22s %8s %8s %8s" % ("文件","总ID","唯一","重复"))
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
    dup = sum(v-1 for v in c.values() if v > 1)
    top = [k for k, v in c.most_common(3) if v > 1]
    print("%-22s %8d %8d %8d  %s" % (tag, len(ids), len(c), dup, ("例: " + str(top)) if top else ""))

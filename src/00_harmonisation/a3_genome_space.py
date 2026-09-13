#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段2b：全量 ID 普查 + 450K 探针→基因映射 + 共同基因空间"""
import os, re, gzip, json, collections
import numpy as np, pandas as pd
TX = "/Volumes/tjogzt4T/data/xena"
CP = "/Volumes/tjogzt4T/data/cptac"

def op_open(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, "rt", errors="replace")

def all_ids(p):
    ids = []
    with op_open(p) as f:
        first = True
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            if first: first = False; continue
            ids.append(line.split("\t", 1)[0].strip().strip('"'))
    return ids

def census(ids):
    c = collections.Counter(); ex = {}
    for v in ids:
        if re.fullmatch(r"\d+", v): k = "Entrez数字"
        elif v.upper().startswith("ENSG"): k = "ENSG"
        elif re.fullmatch(r"cg\d+", v): k = "450K探针"
        elif re.fullmatch(r"hsa-[a-z0-9\-]+", v, re.I): k = "miRNA名"
        elif re.fullmatch(r"MIMAT\d+", v): k = "miRNA(MIMAT)"
        elif re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\._@]*", v): k = "symbol样"
        else: k = "其它/位点ID"
        c[k] += 1; ex.setdefault(k, v)
    return c, ex

TARGETS = {
 "TCGA mRNA(EB++)":   f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
 "TCGA mRNA(Hugo)":   f"{TX}/tcgapancan/tcga_RSEM_Hugo_norm_count.gz",
 "TCGA 450K":         f"{TX}/UCEC/TCGA.UCEC.sampleMap_HumanMethylation450.gz",
 "TCGA RPPA":         f"{TX}/UCEC/TCGA.UCEC.sampleMap_RPPA.gz",
 "TCGA CNA":          f"{TX}/UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
 "CPTAC-ind RNAseq":  f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
 "CPTAC-ind Proteome":f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
 "CPTAC-ind Meth":    f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
 "CPTAC-ind CNA":     f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_WGS_cnv_gistic_thresholded_tumor_v3.0.cct",
 "CPTAC-dis RNAseq":  f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
 "CPTAC-dis Proteome":f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
 "CPTAC-dis Meth":    f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
 "CPTAC-dis SCNV":    f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
 "CPTAC-OV RNAseq":   f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
 "CPTAC-OV Proteome": f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct",
 "CPTAC-OV CNA":      f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_cnv_gene.cct",
}
CEN = {}
IDS = {}
for tag, p in TARGETS.items():
    if not os.path.exists(p): CEN[tag] = ("文件缺失", {}, 0); continue
    ids = all_ids(p); CEN[tag] = (len(ids), dict(census(ids)[0]), census(ids)[1])
    IDS[tag] = set(ids)
    print("%-20s n=%-7d %s" % (tag, len(ids), dict(sorted(census(ids)[0].items(), key=lambda x: -x[1]))))
    print("                     例: %s" % census(ids)[1])
json.dump({k: {"n": v[0], "census": v[1], "examples": v[2]} for k, v in CEN.items()},
          open("/tmp/gyn_retyping/align_ids.json", "w"), indent=1, ensure_ascii=False)

# 450K 探针 → 基因
print("\n=== 450K 探针 → 基因 映射 ===")
am = f"{TX}/tcgapancan/illuminaMethyl450_hg19_GPL16304_TCGAlegacy"
probe2gene = {}
with open(am, errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 2: continue
        g = p[1].strip()
        if g in ("", "."): continue
        probe2gene[p[0].strip()] = [x for x in g.split(",") if x]
probes = IDS.get("TCGA 450K", set())
mapped = [x for x in probes if x in probe2gene]
genes = {g for x in mapped for g in probe2gene[x]}
print("   映射表条目: %d" % len(probe2gene))
print("   TCGA-UCEC 450K 探针: %d，其中有映射: %d (%.1f%%)" % (len(probes), len(mapped), 100*len(mapped)/max(len(probes),1)))
print("   映射到的唯一基因: %d" % len(genes))
print("   示例: cg00651829 -> %s" % probe2gene.get("cg00651829"))
json.dump({"probe2gene_entries": len(probe2gene), "ucec_probes": len(probes),
           "mapped": len(mapped), "genes": len(genes)}, open("/tmp/gyn_retyping/align_probe_map.json","w"), indent=1)

# 共同基因空间（用 symbol 语系）
print("\n=== 共同基因空间（symbol 语系，剔除非 gene-level 层）===")
def symset(tag):
    return {x for x in IDS.get(tag, set()) if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\._@]*", x)}
GROUPS = {
 "表达(RNAseq/mRNA)": ["TCGA mRNA(EB++)", "TCGA mRNA(Hugo)", "CPTAC-ind RNAseq", "CPTAC-dis RNAseq", "CPTAC-OV RNAseq"],
 "蛋白": ["TCGA RPPA", "CPTAC-ind Proteome", "CPTAC-dis Proteome", "CPTAC-OV Proteome"],
 "拷贝数(CNA)": ["TCGA CNA", "CPTAC-ind CNA", "CPTAC-dis SCNV", "CPTAC-OV CNA"],
 "甲基化": ["CPTAC-ind Meth", "CPTAC-dis Meth"],
}
for g, tags in GROUPS.items():
    sets = {t: symset(t) for t in tags if t in IDS}
    sets = {k: v for k, v in sets.items() if v}
    if not sets: continue
    inter = set.intersection(*sets.values())
    print("\n  【%s】" % g)
    for k, v in sets.items(): print("     %-20s %6d" % (k, len(v)))
    print("     >>> 全体交集 = %d" % len(inter))
# 三队列 EC 侧的 symbol 交集
for name, tags in [("三队列 EC 表达交集", ["TCGA mRNA(Hugo)", "CPTAC-ind RNAseq", "CPTAC-dis RNAseq"]),
                   ("三队列 EC 蛋白交集", ["CPTAC-ind Proteome", "CPTAC-dis Proteome"])]:
    s = {t: symset(t) for t in tags if t in IDS}
    if len(s) == len(tags):
        i = set.intersection(*s.values())
        print("\n  %s = %d" % (name, len(i)))

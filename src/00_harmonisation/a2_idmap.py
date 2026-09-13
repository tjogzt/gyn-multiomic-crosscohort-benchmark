#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段2a：盘点本地 ID 映射资源 + 精确判定各文件 ID 类型"""
import os, re, gzip, collections
TX = "/Volumes/tjogzt4T/data/xena"
print("=== 本地 probemap / 注释文件 ===")
hits = []
for root, _, fs in os.walk(TX):
    for f in fs:
        if re.search(r"probemap|annot|gene_id|hugo|gencode|gpl|illumina", f, re.I):
            p = os.path.join(root, f)
            hits.append((os.path.getsize(p), p))
for sz, p in sorted(hits, reverse=True)[:25]:
    print("   %10.1f MB  %s" % (sz/1048576, p.replace(TX, "xena")))
print("   共", len(hits), "个")

def head_rows(path, n=4):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        out = []
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            out.append(line.rstrip("\n").split("\t"))
            if len(out) >= n: break
    return out

print("\n=== probemap 文件结构抽样 ===")
for sz, p in sorted(hits, reverse=True)[:6]:
    try:
        hd = head_rows(p, 4)
        print("\n--- %s" % p.replace(TX, "xena"))
        for r in hd[:4]: print("     ", "\t".join(x[:26] for x in r[:5]))
    except Exception as e:
        print("   ERR", p, e)

print("\n=== 精确判定：抽样 200 个 ID 分类 ===")
FILES = {
 "TCGA-mRNA": f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
 "TCGA-450K": f"{TX}/UCEC/TCGA.UCEC.sampleMap_HumanMethylation450.gz",
 "TCGA-CNA":  f"{TX}/UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
 "CPTAC-ind-RNA": "/Volumes/tjogzt4T/data/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
 "CPTAC-dis-RNA": "/Volumes/tjogzt4T/data/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
}
for tag, p in FILES.items():
    op = gzip.open if p.endswith(".gz") else open
    ids = []
    with op(p, "rt", errors="replace") as f:
        first = True
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            if first: first = False; continue
            ids.append(line.split("\t", 1)[0].strip())
            if len(ids) >= 300: break
    n = len(ids)
    kind = collections.Counter()
    for x in ids:
        v = x.strip('"')
        if re.fullmatch(r"\d+", v): kind["纯数字(Entrez)"] += 1
        elif v.upper().startswith("ENSG"): kind["ENSG"] += 1
        elif v.startswith("cg") and re.fullmatch(r"cg\d+", v): kind["Illumina450K探针"] += 1
        elif re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\._@]*", v): kind["symbol样"] += 1
        else: kind["其它"] += 1
    print("  %-14s n=%d  带引号=%d  %s" % (tag, n, sum(1 for x in ids if x.startswith('"')),
          dict(kind.most_common())))

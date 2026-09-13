#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C-1b：构建跨队列协调化数据层（修正版）"""
import os, gzip, re, json
import numpy as np, pandas as pd, openpyxl
np.random.seed(49)
CP = "/Volumes/tjogzt4T/data/cptac"; TX = "/Volumes/tjogzt4T/data/xena"
OUT = "/tmp/gyn_retyping/harmonized"; os.makedirs(OUT, exist_ok=True)
IDOK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\._@/]*$")

def load(p, cols=None):
    op = gzip.open if p.endswith(".gz") else open
    df = pd.read_csv(p, sep="\t", comment="#", header=0, dtype=str, engine="python", usecols=cols)
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    df = df[df["ID"].str.match(IDOK, na=False)]
    M = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return M.groupby(level=0).mean()

def norm_id(x):
    m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(x).upper().strip())
    return m.group(1) if m else str(x).strip()

# ---------- UCEC 白名单（来自 UCEC 专用 CNA 的列名）----------
ucec = load(f"{TX}/UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz")
UWH = set(norm_id(c) for c in ucec.columns)
print(f"UCEC 白名单: {len(UWH)} 个样本（来自 UCEC 专用 CNA 列名）")

def load_tcga_pancan(path, whitelist):
    """从 pan-cancer 大矩阵按 UCEC 白名单抽列（流式，避免整载）"""
    outp = f"{OUT}/_tmp_{os.path.basename(path)}.pkl"
    if os.path.exists(outp): return pd.read_pickle(outp)
    with gzip.open(path, "rt", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
    keep = [i for i, c in enumerate(hdr) if i > 0 and norm_id(c) in whitelist]
    cols = [0] + keep
    M = load(path, cols=cols)
    M.columns = [norm_id(c) for c in M.columns]
    M = M.T.groupby(level=0).mean().T
    M.to_pickle(outp); return M

print("\n" + "=" * 92); print("【构建】"); print("=" * 92)
INV = {}
def put(coh, lay, M, note=""):
    M = M.dropna(axis=1, how="all")
    INV[f"{coh}|{lay}"] = {"shape": list(M.shape), **({"note": note} if note else {})}
    print(f"  {coh:5s}/{lay:9s} {M.shape[0]:6d} 基因 × {M.shape[1]:4d} 样本  {note}")
    M.to_pickle(f"{OUT}/{coh}__{lay}.pkl")

# ---- TCGA ----
M = load_tcga_pancan(f"{TX}/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz", UWH)
put("TCGA", "mRNA", M, "(pan-cancer EB++ 按 UCEC 白名单)")
M = load(f"{TX}/UCEC/TCGA.UCEC.sampleMap_miRNA_HiSeq_gene.gz"); M.columns = [norm_id(c) for c in M.columns]
put("TCGA", "miRNA", M)
M = ucec.copy(); M.columns = [norm_id(c) for c in M.columns]
put("TCGA", "CNA", M)

# ---- ind ----
def I(path, split=False):
    M = load(path)
    if split:
        cols = {c: I_GRP.get(re.sub(r"-[A-Z]$", "", str(c).upper())) for c in M.columns}
        T = [c for c, g in cols.items() if g == "Tumor"]
        N = [c for c, g in cols.items() if g in ("Adjacent_normal", "Enriched_Normal")]
        return M[T].dropna(axis=1, how="all"), M[N].dropna(axis=1, how="all")
    return M

wb = openpyxl.load_workbook(f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx", read_only=True)
ws = wb[wb.sheetnames[0]]; rows = list(ws.iter_rows(values_only=True)); hdr = [str(x) for x in rows[0]]
gi, ci = hdr.index("Group"), hdr.index("Case_id")
I_GRP = {str(r[ci]).strip().upper(): str(r[gi]) for r in rows[1:] if r[ci]}

B = f"{CP}/CPTAC_UCEC_independent_LinkedOmics"
T, N = I(f"{B}/UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct", True)
put("ind", "mRNA", T); put("ind", "mRNA_N", N)
T, N = I(f"{B}/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct", True)
put("ind", "protein", T); put("ind", "protein_N", N)
T, N = I(f"{B}/UCEC_confirmatory_miRNAseq_miRNA_TPM_log2(x+1)_tumor_normal_v3.0.cct", True)
put("ind", "miRNA", T)
put("ind", "CNA", load(f"{B}/UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct"))
put("ind", "meth", load(f"{B}/UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct"))

# ---- dis ----
B = f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics"
put("dis", "mRNA", load(f"{B}/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct"))
put("dis", "mRNA_N", load(f"{B}/HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Normal.cct"))
put("dis", "protein", load(f"{B}/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct"))
put("dis", "protein_N", load(f"{B}/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Normal.cct"))
put("dis", "miRNA", load(f"{B}/HS_CPTAC_UCEC_microRNA_log2_Tumor.cct"))
put("dis", "CNA", load(f"{B}/HS_CPTAC_UCEC_SCNA_log2_gene_level.cct"))
put("dis", "meth", load(f"{B}/HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct"))

# ---- OV ----
B = f"{CP}/CPTAC_OV_Prospective_LinkedOmics"
put("OV", "mRNA", load(f"{B}/HS_CPTAC_OV_rnaseq_fpkm_log2.cct"))
put("OV", "CNA", load(f"{B}/HS_CPTAC_OV_cnv_gene.cct"))
put("OV", "protein", load(f"{B}/HS_CPTAC_OV_proteome_gene_tumor.cct"))
put("OV", "protein_N", load(f"{B}/HS_CPTAC_OV_proteome_gene_normal.cct"))

# ---- TCGA 甲基化（探针->基因，读缓存）----
import pandas as pd
c = f"{OUT}/TCGA_meth_genes.csv.gz"
if os.path.exists(c):
    M = pd.read_csv(c, index_col=0)
    M.columns = [norm_id(x) for x in M.columns]
    M = M.T.groupby(level=0).mean().T
    put("TCGA", "meth", M, "(450K 探针->基因，本地缓存)")
else:
    print("  !! TCGA/meth 缓存缺失（先跑 e1）")

json.dump(INV, open(f"{OUT}/inventory.json", "w"), indent=1, ensure_ascii=False)
print(f"\n已缓存 {len(INV)} 个层")
for f in os.listdir(OUT):
    if f.startswith("_tmp_"): os.remove(os.path.join(OUT, f))
print("临时文件已清")

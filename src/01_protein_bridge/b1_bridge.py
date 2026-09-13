#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""蛋白层桥接诊断：原始基因均值 vs 队列内 肿瘤−正常 对比"""
import os, gzip, re, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
np.random.seed(49)
CP = "/Volumes/tjogzt4T/data/cptac"

def load(p, max_cols=None):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            cols = line.rstrip("\n").split("\t"); break
    nc = len(cols) - 1
    use = [0] + (list(range(1, nc+1)) if (not max_cols or nc <= max_cols)
                 else sorted(np.random.choice(range(1, nc+1), max_cols, replace=False)))
    df = pd.read_csv(p, sep="\t", comment="#", header=0, usecols=use, dtype=str, engine="python")
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    df = df[df["ID"].str.match(r"^[A-Za-z][A-Za-z0-9\-\._@]*$", na=False)]
    M = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return M.groupby(level=0).mean()

F = {
 "dis_T": f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
 "dis_N": f"{CP}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Normal.cct",
 "ov_T":  f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_tumor.cct",
 "ov_N":  f"{CP}/CPTAC_OV_Prospective_LinkedOmics/HS_CPTAC_OV_proteome_gene_normal.cct",
 "ind_TN":f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
}
M = {}
for k, p in F.items():
    if os.path.exists(p):
        M[k] = load(p); print("%-8s 基因=%-6d 样本=%d" % (k, M[k].shape[0], M[k].shape[1]))
    else:
        print("%-8s 文件缺失" % k)

# ind 的样本分成 tumor/normal：查 meta
tv = None
mp = f"{CP}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx"
if os.path.exists(mp):
    import openpyxl
    wb = openpyxl.load_workbook(mp, read_only=True); ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True)); hdr = [str(x) for x in rows[0]]
    cand = [h for h in hdr if re.search(r"tumor|normal|sample_type|group", h, re.I)]
    print("\nind 的候选分组列:", cand[:6])
    for h in cand:
        vals = [r[hdr.index(h)] for r in rows[1:] if r[hdr.index(h)] is not None]
        from collections import Counter
        c = Counter(str(v) for v in vals)
        if 1 < len(c) <= 6:
            print("   [%s] %s" % (h, c.most_common(6)))
    tv = cand

res = {}
print("\n" + "=" * 96)
print("【诊断 1】原始基因均值谱（肿瘤）")
print("=" * 96)
def prof(m): return m.mean(axis=1, skipna=True)
for a, b in [("dis_T", "ov_T")]:
    if a in M and b in M:
        pa, pb = prof(M[a]), prof(M[b])
        c = pa.index.intersection(pb.index)
        x, y = pa.reindex(c).values, pb.reindex(c).values
        m = np.isfinite(x) & np.isfinite(y)
        rho, pv = spearmanr(x[m], y[m])
        res["raw_%s_%s" % (a, b)] = {"n": int(m.sum()), "rho": float(rho), "p": float(pv)}
        print("   %s × %s  n=%d  rho=%.3f  p=%.1e" % (a, b, m.sum(), rho, pv))

print("\n" + "=" * 96)
print("【诊断 2】队列内 肿瘤−正常 对比（log2 T/N，参考效应在队列内抵消）")
print("=" * 96)
D = {}
for coh, t, n in [("dis", "dis_T", "dis_N"), ("ov", "ov_T", "ov_N")]:
    if t in M and n in M:
        pt, pn = prof(M[t]), prof(M[n])
        c = pt.index.intersection(pn.index)
        D[coh] = (pt.reindex(c) - pn.reindex(c)).dropna()
        print("   %s: 共同基因=%d  Δ 范围[%.2f, %.2f]  中位=%.3f" % (
            coh, len(D[coh]), D[coh].min(), D[coh].max(), D[coh].median()))
if len(D) == 2:
    c = D["dis"].index.intersection(D["ov"].index)
    x, y = D["dis"].reindex(c).values, D["ov"].reindex(c).values
    m = np.isfinite(x) & np.isfinite(y)
    rho, pv = spearmanr(x[m], y[m])
    res["delta_dis_ov"] = {"n": int(m.sum()), "rho": float(rho), "p": float(pv)}
    print("   ★ Δ(dis) × Δ(ov)  n=%d  rho=%.3f  p=%.1e" % (m.sum(), rho, pv))

print("\n" + "=" * 96)
print("【诊断 3】截尾稳健性：剔除均值最极端的 10% 基因后重算")
print("=" * 96)
if "dis_T" in M and "ov_T" in M:
    pa, pb = prof(M["dis_T"]), prof(M["ov_T"])
    c = pa.index.intersection(pb.index)
    df = pd.DataFrame({"a": pa.reindex(c), "b": pb.reindex(c)}).dropna()
    lo, hi = df.stack().quantile([0.05, 0.95])
    sub = df[(df["a"].between(lo, hi)) & (df["b"].between(lo, hi))]
    rho, pv = spearmanr(sub["a"], sub["b"])
    res["trim_dis_ov"] = {"n": len(sub), "rho": float(rho), "p": float(pv)}
    print("   剔除极端后 n=%d  rho=%.3f  p=%.1e" % (len(sub), rho, pv))

json.dump(res, open("/tmp/gyn_retyping/bridge_diag.json", "w"), indent=1, ensure_ascii=False)
print("\n已落盘 bridge_diag.json")

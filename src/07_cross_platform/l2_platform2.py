#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l2：跨平台可比性正式分析
- 确认 Confirmatory 因 median polishing 不可用（有据排除）
- Δ 一致性阶梯 + 置换检验
- 效应量分箱稳健性
- 网络层（蛋白-蛋白相关结构）可比性
- 尺度标定（斜率）
"""
import os, csv, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

csv.field_size_limit(10**9)
BASE = "/Volumes/tjogzt4T/data/cptac"
OUT = "/tmp/gyn_retyping/platform"

def read_cct(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        hdr = next(csv.reader(f, delimiter="\t"))
        rows, idx = [], []
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) != len(hdr): continue
            idx.append(c[0].strip('"')); rows.append([x.strip('"') for x in c[1:]])
    M = pd.DataFrame(rows, index=idx, columns=[h.strip('"') for h in hdr[1:]])
    return M.apply(pd.to_numeric, errors="coerce")

DN = read_cct(f"{BASE}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Normal.cct")
DT = read_cct(f"{BASE}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct")
D = pd.read_csv(f"{OUT}/delta_vectors.csv", index_col=0)
d_dis, d_eeec, d_E, d_L = D["Discovery"], D["EEEC"], D["EEEC_E"], D["EEEC_L"]

# ---------- EEEC 样本级矩阵（肿瘤） ----------
E = "/tmp/gyn_retyping/eeec_batch"
Lm = pd.read_csv(f"{E}/matrix_log2.csv", index_col=0)
meta = pd.read_csv(f"{E}/sample_meta.csv")
meta = meta[meta["family"].isin(["Ecan","Enorm","Lcan","Lnorm"])].reset_index(drop=True)
pgm = pd.read_csv(f"{E}/protein_gene.csv")
gmap = dict(zip(pgm["protein"].astype(str), pgm["gene"].astype(str)))
genesE = np.array([gmap.get(str(p), "") for p in Lm.index])
Et = Lm[[c for c, f in zip(meta["col"], meta["family"]) if f in ("Ecan", "Lcan")]].copy()
Et["__g"] = genesE
Et = Et[Et["__g"] != ""].groupby("__g").mean()
print(f"EEEC 肿瘤矩阵: {Et.shape[0]} 基因 × {Et.shape[1]} 样本")

def perm_test(x, y, n=2000, seed=49):
    rng = np.random.RandomState(seed)
    obs = spearmanr(x, y)[0]
    null = [spearmanr(x, rng.permutation(y))[0] for _ in range(n)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return float(obs), float(p), float(np.mean(np.abs(null)))

common = sorted(set(d_dis.index) & set(d_eeec.index) & set(Et.index) & set(DT.index) & set(DN.index))
print(f"六方公共基因: {len(common)}")
dd = d_dis.reindex(common); de = d_eeec.reindex(common); dE = d_E.reindex(common); dL = d_L.reindex(common)

rows = []
def add(x, y, label, perm=True):
    m = np.isfinite(x.values) & np.isfinite(y.values)
    a, b = x.values[m], y.values[m]
    rho = float(spearmanr(a, b)[0]); r = float(pearsonr(a, b)[0])
    sign = float(np.mean(np.sign(a) == np.sign(b)))
    slope = float(np.polyfit(a, b, 1)[0])
    rec = {"pair": label, "n": int(m.sum()), "spearman": round(rho, 4), "pearson": round(r, 4),
           "sign_conc": round(sign, 4), "slope_EEEC_on_CPTAC": round(slope, 4)}
    if perm:
        _, p, nullabs = perm_test(a, b, 500)
        rec["perm_p"] = p; rec["null_abs_mean"] = round(nullabs, 4)
    rows.append(rec)
    print(f"  {label:38s} n={m.sum():5d}  ρ={rho:+.4f}  r={r:+.4f}  同号={sign*100:5.1f}%  斜率={slope:+.3f}"
          + (f"  置换p={rec['perm_p']:.4f}" if perm else ""))
    return rec

print("\n=== Δ 一致性阶梯 ===")
add(dE, dL, "① EEEC-E vs EEEC-L（同队列跨批次）")
add(de, dd, "③ EEEC vs CPTAC-Discovery")
add(dE, dd, "④ CPTAC-Dis vs EEEC-E")
add(dL, dd, "④ CPTAC-Dis vs EEEC-L")
pd.DataFrame(rows).to_csv(f"{OUT}/platform_ladder_v2.csv", index=False)

# ---------- 效应量分箱稳健性 ----------
print("\n=== 效应量分箱（|Δ_CPTAC-Dis| 分位）===")
q = pd.qcut(dd.abs(), 4, labels=["Q1 最弱", "Q2", "Q3", "Q4 最强"])
bins = []
for lab, idx in dd.groupby(q, observed=True).groups.items():
    idx = pd.Index(idx)
    a = dd.reindex(idx).values; b = de.reindex(idx).values
    m = np.isfinite(a) & np.isfinite(b)
    rho = float(spearmanr(a[m], b[m])[0])
    bins.append({"bin": str(lab), "n": int(m.sum()), "spearman": round(rho, 4),
                 "sign_conc": round(float(np.mean(np.sign(a[m]) == np.sign(b[m]))), 4)})
    print(f"  {str(lab):10s} n={m.sum():5d}  ρ={rho:+.4f}  同号={np.mean(np.sign(a[m])==np.sign(b[m]))*100:.1f}%")
pd.DataFrame(bins).to_csv(f"{OUT}/effect_bins.csv", index=False)

# ---------- 网络层可比性（肿瘤样本内的蛋白-蛋白相关）----------
print("\n=== 网络层可比性 ===")
gn = [g for g in common if g in Et.index and g in DT.index]
Ae = Et.loc[gn].values.astype(float)                    # 基因 × 样本(EEEC 肿瘤 98)
Ad = DT.loc[gn].values.astype(float)                    # 基因 × 样本(CPTAC 肿瘤 95)
Ce = np.corrcoef(Ae); Cd = np.corrcoef(Ad)
iu = np.triu_indices(len(gn), 1)
ve, vd = Ce[iu], Cd[iu]
ok = np.isfinite(ve) & np.isfinite(vd)
net_rho = float(spearmanr(ve[ok], vd[ok])[0])
net_r = float(pearsonr(ve[ok], vd[ok])[0])
print(f"  蛋白对 {ok.sum():,} 个 | 网络结构 Spearman = {net_rho:+.4f} | Pearson = {net_r:+.4f}")
# PC1 载荷对比
from numpy.linalg import svd
def pc1(X):
    X = np.asarray(X, float)
    keep = np.isfinite(X).all(axis=1)          # SVD 要求无缺失
    if keep.sum() < X.shape[0]:
        X = X[keep]
    X = X - X.mean(axis=1, keepdims=True)
    U, s, Vt = svd(X, full_matrices=False)
    v = U[:, 0] * np.sign(U[:, 0].sum())
    return v, keep
p1e, ke = pc1(Ae)
p1d, kd = pc1(Ad)
ie, idd = np.where(ke)[0], np.where(kd)[0]
cpos = np.intersect1d(ie, idd)                       # 全空间中的公共基因位置
if len(cpos) >= 50:
    # 映射到各自“保留子集”中的位置
    pe = np.searchsorted(ie, cpos); pd_ = np.searchsorted(idd, cpos)
    pc1_rho = float(spearmanr(p1e[pe], p1d[pd_])[0])
    print(f"  PC1 载荷一致性 Spearman = {pc1_rho:+.4f}（{len(cpos)} 基因完整）")
else:
    pc1_rho = float("nan"); print("  PC1 可比基因不足")
# 网络层置换
_, np_perm, _ = perm_test(ve[ok], vd[ok], 200)
print(f"  网络结构置换 p = {np_perm:.4f}")
net = {"n_pairs": int(ok.sum()), "network_spearman": round(net_rho, 4),
       "network_pearson": round(net_r, 4), "network_perm_p": np_perm,
       "pc1_loading_spearman": round(pc1_rho, 4), "pc1_n_genes": int(len(cpos)),
       "n_genes": len(gn), "n_tumor_EEEC": int(Ae.shape[1]), "n_tumor_CPTAC": int(Ad.shape[1])}
json.dump({"ladder": rows, "bins": bins, "network": net},
          open(f"{OUT}/platform_results.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 platform_results.json")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l1：EEEC(LFQ) vs CPTAC(TMT) 跨平台可比性
建立效应量阶梯：
  ① 同队列同平台（EEEC E vs L）        —— 上限
  ② 同平台异队列（CPTAC Dis vs Conf）  —— 队列效应
  ③ 异平台异队列（EEEC vs CPTAC）      —— 平台 + 队列
"""
import os, csv, json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

BASE = "/Volumes/tjogzt4T/data/cptac"
OUT = "/tmp/gyn_retyping/platform"
os.makedirs(OUT, exist_ok=True)

def read_cct(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        hdr = next(csv.reader(f, delimiter="\t"))
        rows, idx = [], []
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) != len(hdr): continue
            idx.append(c[0]); rows.append(c[1:])
    M = pd.DataFrame(rows, index=[i.strip('"') for i in idx], columns=[h.strip('"') for h in hdr[1:]])
    return M.apply(pd.to_numeric, errors="coerce")

print("载入 CPTAC Discovery …")
DN = read_cct(f"{BASE}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Normal.cct")
DT = read_cct(f"{BASE}/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct")
print(f"  Discovery Normal {DN.shape} | Tumor {DT.shape}")
print("载入 CPTAC Confirmatory ratio …")
CR = read_cct(f"{BASE}/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct")
print(f"  Confirmatory ratio {CR.shape}")

d_dis = (DT.mean(axis=1) - DN.mean(axis=1)).dropna()
d_conf = CR.mean(axis=1).dropna()
print(f"\nΔ 向量：Discovery {len(d_dis)} 基因 | Confirmatory {len(d_conf)} 基因")

# ---- EEEC Δ ----
E = "/tmp/gyn_retyping/eeec_batch"
Lm = pd.read_csv(f"{E}/matrix_log2.csv", index_col=0)
meta = pd.read_csv(f"{E}/sample_meta.csv")
meta = meta[meta["family"].isin(["Ecan","Enorm","Lcan","Lnorm"])].reset_index(drop=True)
pg = pd.read_csv(f"{E}/protein_gene.csv")
gmap = dict(zip(pg["protein"].astype(str), pg["gene"].astype(str)))
sample_cols = meta["col"].tolist()
X = Lm[sample_cols].T                      # 样本 × 蛋白
X.index = meta["family"].values
genes = np.array([gmap.get(str(p), "") for p in Lm.index])
# 一基因多蛋白：取该基因内均值
def delta_from(sel_t, sel_n):
    T = X.loc[sel_t]; N = X.loc[sel_n]
    d = T.mean(axis=0) - N.mean(axis=0)
    tmp = pd.DataFrame({"g": genes, "d": d.values}).query("g != ''")
    return tmp.groupby("g")["d"].mean()
d_eeec = delta_from(["Ecan","Lcan"], ["Enorm","Lnorm"])
d_eeecE = delta_from(["Ecan"], ["Enorm"])
d_eeecL = delta_from(["Lcan"], ["Lnorm"])
print(f"Δ 向量：EEEC(合并) {len(d_eeec)} | EEEC-E {len(d_eeecE)} | EEEC-L {len(d_eeecL)}")

# ---- 公共基因空间 ----
sets = [set(d_dis.index), set(d_conf.index), set(d_eeec.index), set(d_eeecE.index), set(d_eeecL.index)]
common = set.intersection(*sets)
print(f"\n五者公共基因: {len(common)}")

def pair(a, b, label, boot=2000, seed=49):
    k = sorted(common)
    x = a.reindex(k).values; y = b.reindex(k).values
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    rho, pr = spearmanr(x, y); r, pp = pearsonr(x, y)
    sign = float(np.mean(np.sign(x) == np.sign(y)))
    rng = np.random.RandomState(seed); bs = []
    for _ in range(boot):
        i = rng.randint(0, len(x), len(x))
        bs.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out = {"pair": label, "n": int(len(x)), "spearman": round(float(rho), 4),
           "spearman_lo": round(float(lo), 4), "spearman_hi": round(float(hi), 4),
           "pearson": round(float(r), 4), "sign_concordance": round(sign, 4)}
    print(f"  {label:34s} n={len(x):5d}  Spearman={rho:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"Pearson={r:+.4f}  同号={sign*100:.1f}%")
    return out

print("\n=== 效应量阶梯（Δ = log2 肿瘤 − log2 正常）===")
res = []
res.append(pair(d_eeecE, d_eeecL, "① EEEC-E Δ vs EEEC-L Δ", 500))
res.append(pair(d_dis, d_conf,  "② CPTAC Discovery vs Confirmatory"))
res.append(pair(d_eeec, d_dis,  "③ EEEC vs CPTAC Discovery"))
res.append(pair(d_eeec, d_conf, "③ EEEC vs CPTAC Confirmatory"))
res.append(pair(d_dis, d_eeecE, "④ CPTAC-Dis vs EEEC-E"))
res.append(pair(d_conf, d_eeecL,"④ CPTAC-Conf vs EEEC-L"))

pd.DataFrame(res).to_csv(f"{OUT}/platform_ladder.csv", index=False)
json.dump(res, open(f"{OUT}/platform_ladder.json", "w"), ensure_ascii=False, indent=1)

# 落盘 Δ 向量备后用
pd.DataFrame({"Discovery": d_dis, "Confirmatory": d_conf, "EEEC": d_eeec,
              "EEEC_E": d_eeecE, "EEEC_L": d_eeecL}).to_csv(f"{OUT}/delta_vectors.csv")
print(f"\n已落盘 {OUT}/platform_ladder.csv 与 delta_vectors.csv")

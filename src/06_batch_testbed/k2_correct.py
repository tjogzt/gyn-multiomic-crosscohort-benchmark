#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""k2：EEEC 批次测试床——正式检验「逐基因校正是否足够」
核心判据：把逐基因校正做到理论上限（S2 = 各批次内逐基因 z 标准化，
恰好抹掉全部逐基因均值与方差差异）后，批次可分离性还剩多少。
若仍高 → 批次信号必然在更高阶（协方差/样本级）结构里。
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

OUT = "/tmp/gyn_retyping/eeec_batch"
rng = np.random.RandomState(49)

L = pd.read_csv(f"{OUT}/matrix_log2.csv", index_col=0)      # 蛋白 × 样本
meta = pd.read_csv(f"{OUT}/sample_meta.csv")
meta = meta[meta["family"].isin(["Ecan","Enorm","Lcan","Lnorm"])].reset_index(drop=True)
cols = meta["col"].tolist()
X = L[cols].T.copy()                                        # 样本 × 蛋白
y_batch = (meta["batch"] == "L").astype(int).values
y_cond  = (meta["cond"] == "can").astype(int).values
print(f"矩阵: {X.shape[0]} 样本 × {X.shape[1]} 蛋白 | 批次 L={y_batch.sum()} | 肿瘤={y_cond.sum()}")

# ---------------- 校正臂 ----------------
def s0_raw(X):  return X.values

def s1_zscore(X):                     # 全局逐基因 z
    Z = (X - X.mean()) / X.std(ddof=1).replace(0, 1)
    return Z.values

def s2_zscore_within_batch(X):        # 各批次内逐基因 z（逐基因校正的理论上限）
    Z = X.copy()
    for b in [0, 1]:
        sub = X.iloc[y_batch == b]
        Z.iloc[y_batch == b] = (sub - sub.mean()) / sub.std(ddof=1).replace(0, 1)
    return Z.values

def s3_quantile(X):                   # 逐样本分位数标准化到共同参考
    R = X.values.copy()
    ref = np.sort(X.values, axis=1).mean(axis=0)            # 每个样本内排序后按秩平均 → 长度 = 蛋白数
    for j in range(R.shape[0]):
        order = np.argsort(R[j])
        R[j, order] = ref
    return R

def s4_rankint(X):                    # 逐样本秩 → 逆正态
    from scipy.stats import norm
    R = X.values.copy(); n = R.shape[1]
    for j in range(R.shape[0]):
        order = np.argsort(R[j]); r = np.empty(n); r[order] = np.arange(1, n+1)
        R[j] = norm.ppf((r - 0.5) / n)
    return R

def s5_harmony(X, npc=50):            # 嵌入层校正（协方差层）
    import harmonypy
    Xs = StandardScaler().fit_transform(X.values)
    P = PCA(n_components=npc, random_state=49).fit(Xs)
    ho = harmonypy.run_harmony(P.transform(Xs), meta_batch, ["batch"], max_iter_harmony=20)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != X.shape[0]: Z = Z.T          # 朝向随版本而异，按样本数对齐
    return Z

meta_batch = meta[["batch"]].copy()
meta_batch["batch"] = meta_batch["batch"].astype(str)

ARMS = {
    "S0 原始 log2":            ("raw",        s0_raw),
    "S1 逐基因 z（全局）":       ("pygen",      s1_zscore),
    "S2 批次内逐基因 z（上限）":  ("pygen_max",  s2_zscore_within_batch),
    "S3 逐样本分位数标准化":      ("quantile",   s3_quantile),
    "S4 逐样本秩→逆正态":        ("rankint",    s4_rankint),
    "S5 Harmony（嵌入层）":      ("embed",      s5_harmony),
}

M = {}
for name, (tag, fn) in ARMS.items():
    try:
        V = np.asarray(fn(X), dtype=float)
        V[~np.isfinite(V)] = 0.0
        M[name] = V
        print(f"  ✓ {name:24s} {V.shape}")
    except Exception as e:
        print(f"  ✗ {name:24s} {type(e).__name__}: {str(e)[:90]}")

# 导出 R 侧所需
X.to_csv(f"{OUT}/for_R_matrix.csv")
pd.DataFrame({"batch": meta["batch"].values, "cond": meta["cond"].values}).to_csv(
    f"{OUT}/for_R_meta.csv", index=False)
np.save(f"{OUT}/y_batch.npy", y_batch); np.save(f"{OUT}/y_cond.npy", y_cond)
json.dump({k: v.tolist() for k, v in M.items()} if False else
          {k: [list(v.shape)] for k, v in M.items()},
          open(f"{OUT}/arms_py.json", "w"), ensure_ascii=False, indent=1)
np.savez_compressed(f"{OUT}/arms_py.npz", **{k: v for k, v in M.items()})
print(f"\n已落盘 Python 臂 {len(M)} 个 → arms_py.npz；并导出 for_R_*.csv 供 ComBat/limma")

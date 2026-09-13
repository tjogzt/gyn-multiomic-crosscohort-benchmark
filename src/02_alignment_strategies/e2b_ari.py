#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C-2b：对齐策略 × 层 × 队列对 的迁移 ARI 矩阵（修正版）
设计：所有策略输出「可直接聚类的 样本×基因 矩阵」，主循环不再二次标准化
  S0 基线        = 逐样本中位数居中 + 逐基因 z-score（队列内）
  S1 分位数标准化 = S0 之前加逐样本分位数标准化
  S2 Δ参考校正   = 逐基因减去本队列正常样本均值，再 z-score（需正常样本）
  S3 ComBat(L/S) = 把 A、B 池化为批次做位置/尺度校正，再 z-score
"""
import os, json, itertools, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings("ignore")
np.random.seed(49)
H = "/tmp/gyn_retyping/harmonized"
OUT = "/tmp/gyn_retyping/ari_matrix.json"
NGENE, KGRID = 2000, [3, 4, 5]

def L(coh, lay):
    p = f"{H}/{coh}__{lay}.pkl"
    return pd.read_pickle(p) if os.path.exists(p) else None

def medcenter(X):
    return X - np.nanmedian(X, axis=1, keepdims=True)

def qnorm(X):
    """逐样本分位数标准化：把每样本排序后的值映射到参考分位数（按排序序）"""
    A = np.array(X, dtype=float)
    ref = np.nanmean(np.sort(A, axis=1), axis=0)
    n = len(ref)
    for i in range(A.shape[0]):
        r = A[i]; ok = np.isfinite(r); k = int(ok.sum())
        if k < 3: continue
        idx = np.where(ok)[0]
        order = np.argsort(r[idx])                 # 样本内升序位置
        vals = np.interp(np.linspace(0, 1, k), np.linspace(0, 1, n), ref)
        r2 = r.copy()
        r2[idx[order]] = vals                      # ★ 按排序序赋回
        A[i] = r2
    return A

def zgene(X):
    m = np.nanmean(X, axis=0, keepdims=True)
    s = np.nanstd(X, axis=0, keepdims=True)
    s = np.where((s < 1e-9) | ~np.isfinite(s), 1.0, s)
    Z = (X - m) / s
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

def combat_ls(Ya, Yb):
    """无经验贝叶斯的 ComBat（位置/尺度校正），批次 = 队列"""
    Y = np.vstack([Ya, Yb])
    batch = np.array([0] * len(Ya) + [1] * len(Yb))
    gm = Y.mean(axis=0); gv = Y.var(axis=0)
    gv[gv < 1e-12] = 1.0
    Z = np.zeros_like(Y)
    for b in (0, 1):
        m = batch == b
        bm = Y[m].mean(axis=0); bv = Y[m].var(axis=0)
        bv[bv < 1e-12] = 1.0
        Z[m] = (Y[m] - bm) / np.sqrt(bv)
    # 校正后补回总体位置/尺度（可选：这里只保尺度，避免注入总体均值造成队列间假相似）
    Z = Z * np.sqrt(gv)
    return Z[:len(Ya)], Z[len(Ya):]

def top_mad(X, ng=NGENE):
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    mad = np.where(np.isfinite(mad), mad, -1)
    return set(np.argsort(-mad)[:ng].tolist())

def transfer_ari(Xa, Xb, K, seed=49):
    """A 上聚类 → B 样本按最近质心指派 → 与 B 自身聚类比 ARI（两边同为 B 样本）"""
    kmA = KMeans(K, n_init=20, random_state=seed).fit(Xa)
    kmB = KMeans(K, n_init=20, random_state=seed).fit(Xb)
    C = kmA.cluster_centers_
    pred = ((Xb[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    # 伪解检测：某一臂退化为单簇则 ARI 无意义
    deg = (len(set(kmB.labels_)) < 2) or (len(set(pred)) < 2) or (len(set(kmA.labels_)) < 2)
    return float(adjusted_rand_score(kmB.labels_, pred)), bool(deg)

LAYERS = {
 "mRNA":   ["TCGA", "ind", "dis", "OV"],
 "miRNA":  ["TCGA", "ind", "dis"],
 "CNA":    ["TCGA", "ind", "dis", "OV"],
 "protein":["ind", "dis", "OV"],
 "meth":   ["TCGA", "ind", "dis"],
}
NORMALS = {("ind","mRNA"):"mRNA_N", ("dis","mRNA"):"mRNA_N",
           ("ind","protein"):"protein_N", ("dis","protein"):"protein_N", ("OV","protein"):"protein_N"}

res = []
print("=" * 104)
print(f"{'层':<9}{'策略':<16}{'队列对':<14}{'n(A/B)':>11}{'基因':>6}    ARI(K=3/4/5)      退化")
print("=" * 104)
for lay, cohs in LAYERS.items():
    RAW = {c: L(c, lay) for c in cohs}
    RAW = {c: v for c, v in RAW.items() if v is not None}
    NRM = {c: (L(c, NORMALS[(c, lay)]) if (c, lay) in NORMALS else None) for c in RAW}
    for a, b in itertools.combinations(sorted(RAW), 2):
        A, B = RAW[a], RAW[b]
        An, Bn = NRM[a], NRM[b]
        g = A.index.intersection(B.index)
        if len(g) < 400:
            res.append({"layer": lay, "pair": f"{a}×{b}", "ari": None, "reason": f"共同基因不足({len(g)})"}); continue
        A0 = medcenter(A.loc[g].values.T.astype(float))
        B0 = medcenter(B.loc[g].values.T.astype(float))
        A0 = np.nan_to_num(A0, nan=0.0); B0 = np.nan_to_num(B0, nan=0.0)
        # 高变基因：在两队列处理后数据上取 MAD top 的交集
        keep = np.array(sorted(top_mad(A0) & top_mad(B0)), dtype=int)
        if len(keep) < 100:
            res.append({"layer": lay, "pair": f"{a}×{b}", "ari": None, "reason": "高变基因不足"}); continue
        A0k, B0k = A0[:, keep], B0[:, keep]
        arms = {"S0_基线": (A0k, B0k)}
        arms["S1_分位数标准化"] = (qnorm(A0k), qnorm(B0k))
        if An is not None and Bn is not None:
            ran = np.nanmean(An.reindex(g).values[keep, :].astype(float), axis=1)   # 每基因正常均值
            rbn = np.nanmean(Bn.reindex(g).values[keep, :].astype(float), axis=1)
            arms["S2_Δ参考校正"] = (np.nan_to_num(A0k - ran, nan=0.0), np.nan_to_num(B0k - rbn, nan=0.0))
        ca, cb = combat_ls(A0k, B0k)
        arms["S3_ComBat(池化)"] = (ca, cb)
        for sname, (Xa, Xb) in arms.items():
            Xa, Xb = zgene(Xa), zgene(Xb)
            aris, degs = {}, {}
            for K in KGRID:
                try:
                    v, d = transfer_ari(Xa, Xb, K)
                    aris[K] = round(v, 4); degs[K] = d
                except Exception as ex:
                    aris[K] = None; degs[K] = None
            res.append({"layer": lay, "strategy": sname, "pair": f"{a}×{b}",
                        "nA": int(A0k.shape[0]), "nB": int(B0k.shape[0]), "ngene": int(len(keep)),
                        "ari": aris, "degenerate": degs})
            print(f"{lay:<9}{sname:<16}{a+'×'+b:<14}{str(A0k.shape[0])+'/'+str(B0k.shape[0]):>11}"
                  f"{len(keep):>6}    {aris.get(3)}/{aris.get(4)}/{aris.get(5)}"
                  f"      {sum(1 for v in degs.values() if v)}/3")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("\n已落盘", OUT)

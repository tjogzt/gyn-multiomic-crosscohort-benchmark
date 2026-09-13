#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""k5：复验批次可分离性（不依赖分类器符号的确定性判据）
1) centroid_AUC：投影到 (均值_E − 均值_L) 方向的 AUC（固定符号，直接度量逐基因均值分离）
2) max_auc：CV AUC 的 max(a, 1−a)（排除符号翻转造成的 0.0000 假象）
3) KS 中位：逐蛋白 E vs L 的 KS 统计量中位数（分布层，不看均值）
4) knn_mix：局部邻域混合度（0.5 = 完全混合）
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

D = "/tmp/gyn_retyping/eeec_batch"
yb = np.load(f"{D}/y_batch.npy"); yc = np.load(f"{D}/y_cond.npy")
meta = pd.read_csv(f"{D}/sample_meta.csv"); meta = meta[meta["family"].isin(["Ecan","Enorm","Lcan","Lnorm"])].reset_index(drop=True)

A = {k: v for k, v in np.load(f"{D}/arms_py.npz").items()}
for nm, f in {"ComBat 规范（sva）":"arm_ComBat.csv", "ComBat mean-only":"arm_ComBatMeanOnly.csv",
              "removeBatchEffect（limma）":"arm_removeBatchEffect.csv"}.items():
    p = os.path.join(D, f)
    if os.path.exists(p): A[nm] = pd.read_csv(p, index_col=0).values

ORDER = ["S0 原始 log2", "S1 逐基因 z（全局）", "S2 批次内逐基因 z（上限）",
         "S3 逐样本分位数标准化", "S4 逐样本秩→逆正态",
         "ComBat 规范（sva）", "ComBat mean-only", "removeBatchEffect（limma）", "S5 Harmony（嵌入层）"]

def clean(V):
    V = np.asarray(V, float).copy(); V[~np.isfinite(V)] = 0.0; return V

def centroid_auc(V, y):
    V = clean(V)
    m1 = V[y == 1].mean(axis=0); m0 = V[y == 0].mean(axis=0)
    d = m1 - m0
    if np.allclose(d, 0): return 0.5
    return float(roc_auc_score(y, V @ d))

def cv_auc_both(V, y, seed=49, clf="lr"):
    V = clean(V)
    est = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
           if clf == "lr" else make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=20000)))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    if clf == "lr":
        p = cross_val_predict(est, V, y, cv=cv, method="predict_proba")[:, 1]
    else:
        p = cross_val_predict(est, V, y, cv=cv, method="decision_function")
    a = float(roc_auc_score(y, p)); return a, max(a, 1 - a)

def ks_median(V):
    V = clean(V)
    return float(np.median([ks_2samp(V[yb == 1, j], V[yb == 0, j]).statistic for j in range(V.shape[1])]))

def knn_mix(V, k=10):
    V = clean(V); V = StandardScaler().fit_transform(V)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(V)
    _, idx = nn.kneighbors(V)
    return float(np.mean(yb[idx[:, 1:]] != yb[:, None]))

rows = []
print(f"{'臂':28s} {'centroid_AUC':>12s} {'LR_AUC':>8s} {'max(AUC,1-A)':>12s} {'SVM_max':>8s} {'KS中位':>8s} {'knn_mix':>8s}")
for k in ORDER:
    if k not in A: continue
    V = A[k]
    ca = centroid_auc(V, yb)
    r_, ma = cv_auc_both(V, yb, clf="lr")
    _, ms = cv_auc_both(V, yb, clf="svm")
    km = ks_median(V); kn = knn_mix(V)
    rows.append({"arm": k, "centroid_AUC": round(ca, 4), "LR_AUC_raw": round(r_, 4),
                 "LR_AUC_abs": round(ma, 4), "SVM_AUC_abs": round(ms, 4),
                 "KS_median": round(km, 4), "knn_mix": round(kn, 4)})
    print(f"{k:28s} {ca:12.4f} {r_:8.4f} {ma:12.4f} {ms:8.4f} {km:8.4f} {kn:8.4f}")

T = pd.DataFrame(rows); T.to_csv(f"{D}/batch_verify.csv", index=False)
json.dump(T.to_dict(orient="records"), open(f"{D}/batch_verify.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 batch_verify.json")

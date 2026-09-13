#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""k4：EEEC 批次测试床——指标计算
每臂算：批次可分离 AUC（越低越好）、生物学可分离 AUC（越高越好）、
跨批次标签迁移准确率（越高越好）、批内-批间相关差（越低越好）、kNN 批次混合度（越高越好）
外加协方差层检验：S2（逐基因上限）再删前 k 个主成分的扫描
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

D = "/tmp/gyn_retyping/eeec_batch"
y_batch = np.load(f"{D}/y_batch.npy"); y_cond = np.load(f"{D}/y_cond.npy")
meta = pd.read_csv(f"{D}/sample_meta.csv"); meta = meta[meta["family"].isin(["Ecan","Enorm","Lcan","Lnorm"])].reset_index(drop=True)
names = list(meta["family"].values)

# ---------- 载入所有臂 ----------
A = {}
z = np.load(f"{D}/arms_py.npz")
for k in z.files: A[k] = z[k]
PY_NAME = {"S0 原始 log2":"raw", "S1 逐基因 z（全局）":"pygen", "S2 批次内逐基因 z（上限）":"pygen_max",
           "S3 逐样本分位数标准化":"quantile", "S4 逐样本秩→逆正态":"rankint", "S5 Harmony（嵌入层）":"embed"}
R_NAME = {"ComBat 规范（sva）":"arm_ComBat.csv", "ComBat mean-only":"arm_ComBatMeanOnly.csv",
          "removeBatchEffect（limma）":"arm_removeBatchEffect.csv"}
for nm, f in R_NAME.items():
    p = os.path.join(D, f)
    if os.path.exists(p):
        A[nm] = pd.read_csv(p, index_col=0).values

ORDER = ["S0 原始 log2", "S1 逐基因 z（全局）", "S2 批次内逐基因 z（上限）",
         "S3 逐样本分位数标准化", "S4 逐样本秩→逆正态",
         "ComBat 规范（sva）", "ComBat mean-only", "removeBatchEffect（limma）", "S5 Harmony（嵌入层）"]

# ---------- 指标 ----------
def auc_cv(V, y, seed=49):
    V = np.asarray(V, float); V[~np.isfinite(V)] = 0
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, V, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))

def transfer_acc(V, y_cond, y_batch):
    """在 E 批次上训练 can/norm，在 L 批次上测；反向同做，取平均"""
    V = np.asarray(V, float); V[~np.isfinite(V)] = 0
    V = StandardScaler().fit_transform(V)
    accs = []
    for tr, te in [(0, 1), (1, 0)]:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        m.fit(V[y_batch == tr], y_cond[y_batch == tr])
        accs.append(float((m.predict(V[y_batch == te]) == y_cond[y_batch == te]).mean()))
    return float(np.mean(accs)), accs

def corr_gap(V):
    C = np.corrcoef(np.asarray(V, float))
    iu = np.triu_indices(len(C), 1)
    same = (y_batch[iu[0]] == y_batch[iu[1]])
    return float(C[iu][same].mean() - C[iu][~same].mean())

def knn_mix(V, k=10):
    V = np.asarray(V, float); V[~np.isfinite(V)] = 0
    V = StandardScaler().fit_transform(V)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(V)
    _, idx = nn.kneighbors(V)
    return float(np.mean(y_batch[idx[:, 1:]] != y_batch[:, None]))

rows = []
for nm in ORDER:
    if nm not in A: continue
    V = A[nm]
    r = {"arm": nm, "n_feat": int(V.shape[1]),
         "batch_AUC": round(auc_cv(V, y_batch), 4),
         "bio_AUC":   round(auc_cv(V, y_cond), 4)}
    ta, tl = transfer_acc(V, y_cond, y_batch)
    r["transfer_acc"] = round(ta, 4); r["transfer_E2L"] = round(tl[0], 4); r["transfer_L2E"] = round(tl[1], 4)
    r["corr_gap"] = round(corr_gap(V), 4)
    r["knn_mix"] = round(knn_mix(V), 4)
    rows.append(r)
T = pd.DataFrame(rows)
print("=== 批次校正臂全指标 ===")
print(T.to_string(index=False))
T.to_csv(f"{D}/batch_metrics.csv", index=False)

# ---------- 协方差层检验：S2 再删前 k 个 PC ----------
print("\n=== 协方差层检验：S2（逐基因上限）再删前 k 个主成分 ===")
S2 = A["S2 批次内逐基因 z（上限）"]
S2s = StandardScaler().fit_transform(S2)
P = PCA(n_components=min(30, S2s.shape[0] - 1), random_state=49).fit(S2s)
Z = P.transform(S2s)
sweep = []
for k in [0, 1, 2, 3, 4, 5, 8, 12, 20]:
    Vk = Z[:, k:] if k < Z.shape[1] else Z[:, :1]
    sweep.append({"k_removed": k,
                  "batch_AUC": round(auc_cv(Vk, y_batch), 4),
                  "bio_AUC": round(auc_cv(Vk, y_cond), 4),
                  "transfer_acc": round(transfer_acc(Vk, y_cond, y_batch)[0], 4),
                  "corr_gap": round(corr_gap(Vk), 4)})
SW = pd.DataFrame(sweep)
print(SW.to_string(index=False))
SW.to_csv(f"{D}/pc_sweep.csv", index=False)

# PC 与批次的关联
print("\n=== 各主成分与批次的关联（|AUROC|）与解释方差 ===")
assoc = []
for i in range(10):
    a = float(roc_auc_score(y_batch, Z[:, i]))
    assoc.append({"PC": i + 1, "AUROC_vs_batch": round(max(a, 1 - a), 4),
                  "AUROC_vs_cond": round(max(float(roc_auc_score(y_cond, Z[:, i])), 1 - float(roc_auc_score(y_cond, Z[:, i]))), 4),
                  "var_exp": round(float(P.explained_variance_ratio_[i]), 4)})
AС = pd.DataFrame(assoc); print(AС.to_string(index=False))
AС.to_csv(f"{D}/pc_assoc.csv", index=False)

json.dump({"metrics": T.to_dict(orient="records"), "pc_sweep": SW.to_dict(orient="records"),
           "pc_assoc": AС.to_dict(orient="records")},
          open(f"{D}/batch_results.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 batch_results.json")

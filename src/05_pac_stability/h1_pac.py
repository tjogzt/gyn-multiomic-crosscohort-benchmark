#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-7：bootstrap PAC 稳定性（P1 硬项）
PAC = 共识矩阵中落在 (0.1, 0.9) 的样本对比例（越低越稳）
对每个 (队列 × 层子集 × 方法) 计算：队列内聚类的稳定性
"""
import os, sys, json, re, itertools, time, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
np.random.seed(49)

H = "/tmp/gyn_retyping/hcsv"
OUT = "/tmp/gyn_retyping/pac_matrix.json"
NB, K = 12, 4          # bootstrap 次数 / 簇数
NGENE = 1500

def L(coh, lay):
    p = f"{H}/{coh}__{lay}.csv.gz"
    if not os.path.exists(p): return None
    M = pd.read_csv(p, index_col=0); M.columns = [str(c) for c in M.columns]; return M

def zgene(X):
    m = np.nanmean(X, axis=0, keepdims=True); s = np.nanstd(X, axis=0, keepdims=True)
    s = np.where((s < 1e-9) | ~np.isfinite(s), 1.0, s)
    return np.nan_to_num((X - m) / s, nan=0.0, posinf=0.0, neginf=0.0)

def topmad_k(X, ng):
    med = np.median(X, axis=0); mad = np.median(np.abs(X - med), axis=0)
    k = np.argsort(-mad)[:ng]; return k[mad[k] > 0]

# ---- 与 f2_benchmark.py 一致的方法特征空间 ----
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances
def knn_affinity(X, k):
    D = euclidean_distances(X); nz = D[D > 0]; mu = np.median(nz) if len(nz) else 1.0
    S = np.exp(-D ** 2 / (2 * mu ** 2)); n = S.shape[0]; K_ = np.zeros_like(S)
    for i in range(n):
        idx = np.argsort(-S[i])[:k]; K_[i, idx] = S[i, idx]
    K_ = (K_ + K_.T) / 2; rs = K_.sum(1, keepdims=True); rs[rs < 1e-12] = 1
    return K_ / rs

def m_kmeans_concat(ls, K): return np.hstack(ls)
def m_nmf_concat(ls, K):
    X = np.hstack(ls); X = X - X.min(0, keepdims=True) + 1e-3
    return NMF(n_components=min(10, X.shape[1]-1), init="nndsvd", random_state=49, max_iter=400).fit_transform(X)
def m_pca_concat(ls, K):
    X = np.hstack(ls); return PCA(n_components=min(10, X.shape[0]-1, X.shape[1]), random_state=49).fit_transform(X)
def m_spec_concat(ls, K):
    X = np.hstack(ls); n = X.shape[0]
    return SpectralEmbedding(n_components=min(8, n-1), affinity="nearest_neighbors",
                             n_neighbors=max(5, min(15, n//5)), random_state=49).fit_transform(X)
def m_mcca_lite(ls, K):
    return np.hstack([PCA(n_components=min(5, X.shape[0]-1, X.shape[1]), random_state=49).fit_transform(X) for X in ls])
def m_mofa_lite(ls, K):
    outs, ws = [], []
    for X in ls:
        p = PCA(n_components=min(5, X.shape[0]-1, X.shape[1]), random_state=49).fit(X)
        outs.append(p.transform(X)); ws.append(max(float(p.explained_variance_ratio_.sum()), 1e-6))
    ws = np.array(ws); ws /= ws.sum()
    return np.hstack([o * np.sqrt(w) for o, w in zip(outs, ws)])
def m_consensus(ls, K):
    outs = []
    for X in ls:
        try:
            lb = KMeans(K, n_init=10, random_state=49).fit_predict(X)
            ind = np.zeros((len(lb), K)); ind[np.arange(len(lb)), lb] = 1.0
            outs.append(ind)
        except Exception: outs.append(np.zeros((X.shape[0], K)))
    return np.hstack(outs)
def m_snf(ls, K, t=20):
    n = ls[0].shape[0]
    if n < 15: return ls[0]
    k = max(3, min(20, n // 10))
    W = [knn_affinity(X, k) for X in ls]; P = [w.copy() for w in W]
    for _ in range(t):
        newP = []
        for i in range(len(P)):
            o = np.zeros_like(P[i])
            for j in range(len(P)):
                if j != i: o = o + P[j]
            o /= max(len(P) - 1, 1); newP.append(W[i] @ o @ W[i].T)
        P = [(p + p.T) / 2 for p in newP]
    Sf = np.mean(P, axis=0); Sf = np.clip((Sf + Sf.T) / 2, 0, None)
    d = min(8, n - 2)
    try: return SpectralEmbedding(n_components=d, affinity="precomputed", random_state=49).fit_transform(Sf)
    except Exception: return np.hstack(ls)

METHODS = {
 "1_SNF": lambda ls, K: m_snf(ls, K),
 "2_NMF(concat)": lambda ls, K: m_nmf_concat(ls, K),
 "3_KMeans(concat)": lambda ls, K: m_kmeans_concat(ls, K),
 "4_PCA(concat)": lambda ls, K: m_pca_concat(ls, K),
 "5_谱嵌入(concat)": lambda ls, K: m_spec_concat(ls, K),
 "6_共联共识": lambda ls, K: m_consensus(ls, K),
 "7_MCCA-lite": lambda ls, K: m_mcca_lite(ls, K),
 "8_MOFA-lite": lambda ls, K: m_mofa_lite(ls, K),
}

def pac(X, K=K, nboot=NB, seed=49):
    """bootstrap 共识矩阵 → PAC；同时返回强共识比例与最大簇占比"""
    rng = np.random.RandomState(seed); n = X.shape[0]
    if n < 40: return None
    acc = np.zeros((n, n)); cnt = np.zeros((n, n))
    ok = 0
    for _ in range(nboot):
        idx = rng.choice(n, n, replace=True); uniq = np.unique(idx)
        if len(uniq) < K * 3: continue
        try: lb = KMeans(K, n_init=5, random_state=seed).fit_predict(X[uniq])
        except Exception: continue
        if len(set(lb)) < 2: continue
        present = np.zeros(n, bool); present[uniq] = True
        lb_full = np.full(n, -1); lb_full[uniq] = lb
        same = (lb_full[:, None] == lb_full[None, :]) & present[:, None] & present[None, :]
        acc += same; cnt += present[:, None] & present[None, :]
        ok += 1
    if ok < 4: return None
    with np.errstate(invalid="ignore", divide="ignore"):
        cons = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    iu = np.triu_indices(n, 1)
    v = cons[iu]; m = np.isfinite(v)
    if m.sum() < 100: return None
    v = v[m]
    return {"pac": float(np.mean((v > 0.1) & (v < 0.9))),
            "strong": float(np.mean(v > 0.9)),
            "weak": float(np.mean(v < 0.1)),
            "median_cons": float(np.median(v)),
            "boot_ok": ok}

DOMAINS = {
 "A_CPTAC三队列": {"cohorts": ["ind", "dis", "OV"], "layers": ["mRNA", "CNA", "protein"]},
 "B_EC四层": {"cohorts": ["TCGA", "ind", "dis"], "layers": ["mRNA", "miRNA", "CNA", "meth"]},
}

res = []
t0 = time.time()
CACHE = {}
for dname, D in DOMAINS.items():
    cohs, lays = D["cohorts"], D["layers"]
    print(f"\n{'='*100}\n域 {dname}  {cohs} × {lays}\n{'='*100}", flush=True)
    for c in cohs:
        for r in range(1, len(lays)+1):
            for sub in itertools.combinations(lays, r):
                Ms = {}
                for l in sub:
                    key = (c, l)
                    if key not in CACHE: CACHE[key] = L(c, l)
                    if CACHE[key] is None: Ms = None; break
                    Ms[l] = CACHE[key]
                if not Ms: continue
                # 样本交集
                s = None
                for M in Ms.values():
                    s = set(M.columns) if s is None else (s & set(M.columns))
                s = sorted(s)
                if len(s) < 40: continue
                blocks = []
                ok = True
                for l in sub:
                    M = Ms[l][s]
                    gk = M.index
                    X = zgene(M.values.T.astype(float))
                    k = topmad_k(X, NGENE)
                    if len(k) < 50: k = np.arange(min(NGENE, X.shape[1]))
                    blocks.append(X[:, k])
                for mname, fn in METHODS.items():
                    try:
                        F = zgene(np.nan_to_num(np.asarray(fn(blocks, K), float), nan=0.0))
                    except Exception as e:
                        res.append({"domain": dname, "cohort": c, "subset": "+".join(sub),
                                    "method": mname, "pac": None, "err": str(e)[:60]}); continue
                    if F.shape[0] < 40:
                        res.append({"domain": dname, "cohort": c, "subset": "+".join(sub),
                                    "method": mname, "pac": None, "err": "n<40"}); continue
                    p = pac(F)
                    rec = {"domain": dname, "cohort": c, "subset": "+".join(sub), "method": mname,
                           "n": int(F.shape[0]), "nfeat": int(F.shape[1])}
                    rec.update(p if p else {"pac": None})
                    res.append(rec)
                json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
                print(f"  {c:6s} {'+'.join(sub):<26} n={len(s):4d} ({time.time()-t0:.0f}s, {len(res)} 条)", flush=True)
print(f"\n完成 {len(res)} 条，用时 {time.time()-t0:.0f}s")
print("已落盘", OUT)

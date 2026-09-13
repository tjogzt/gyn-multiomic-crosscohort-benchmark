#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-1b：多组学整合基准（8 方法 × 逐层子集 × 队列对 的迁移 ARI）
域A：CPTAC 三队列 ind/dis/OV，层 mRNA+CNA+protein（7 个非空子集）
域B：TCGA+ind+dis，层 mRNA+miRNA+CNA+meth（15 个非空子集）
"""
import os, json, itertools, warnings, time
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
warnings.filterwarnings("ignore")
np.random.seed(49)
H = "/tmp/gyn_retyping/harmonized"
OUT = "/tmp/gyn_retyping/bench_matrix.json"
NGENE, KGRID, NBOOT = 1500, [3, 4, 5], 12

def L(coh, lay):
    p = f"{H}/{coh}__{lay}.pkl"
    return pd.read_pickle(p) if os.path.exists(p) else None

def zgene(X):
    m = np.nanmean(X, axis=0, keepdims=True)
    s = np.nanstd(X, axis=0, keepdims=True)
    s = np.where((s < 1e-9) | ~np.isfinite(s), 1.0, s)
    return np.nan_to_num((X - m) / s, nan=0.0, posinf=0.0, neginf=0.0)

def topmad_k(X, ng):
    med = np.median(X, axis=0); mad = np.median(np.abs(X - med), axis=0)
    k = np.argsort(-mad)[:ng]
    return k[mad[k] > 0]

def knn_affinity(X, k):
    D = euclidean_distances(X)
    mu = np.median(D[D > 0]) if (D > 0).any() else 1.0
    S = np.exp(-D ** 2 / (2 * mu ** 2))
    n = S.shape[0]
    K = np.zeros_like(S)
    for i in range(n):
        idx = np.argsort(-S[i])[:k]
        K[i, idx] = S[i, idx]
    K = (K + K.T) / 2
    rs = K.sum(axis=1, keepdims=True); rs[rs < 1e-12] = 1
    return K / rs

# ---------------- 8 种整合方法：每层 → 特征块，返回 样本×特征 ----------------
def m_kmeans_concat(ls, K):     return np.hstack(ls)
def m_nmf_concat(ls, K):
    X = np.hstack(ls); X = X - X.min(0, keepdims=True) + 1e-3
    return NMF(n_components=min(10, X.shape[1]-1), init="nndsvd", random_state=49, max_iter=400).fit_transform(X)
def m_pca_concat(ls, K):
    X = np.hstack(ls); return PCA(n_components=min(10, X.shape[0]-1), random_state=49).fit_transform(X)
def m_spec_concat(ls, K):
    X = np.hstack(ls)
    return SpectralEmbedding(n_components=min(8, X.shape[0]-1), affinity="nearest_neighbors",
                             n_neighbors=max(5, min(15, X.shape[0]//5)), random_state=49).fit_transform(X)
def m_mcca_lite(ls, K):
    return np.hstack([PCA(n_components=min(5, X.shape[0]-1), random_state=49).fit_transform(X) for X in ls])
def m_mofa_lite(ls, K):
    outs, ws = [], []
    for X in ls:
        p = PCA(n_components=min(5, X.shape[0]-1), random_state=49).fit(X)
        outs.append(p.transform(X)); ws.append(max(float(p.explained_variance_ratio_.sum()), 1e-6))
    ws = np.array(ws); ws /= ws.sum()
    return np.hstack([o * np.sqrt(w) for o, w in zip(outs, ws)])
def m_consensus(ls, K):
    n = ls[0].shape[0]; C = np.zeros((n, n)); m = 0
    for X in ls:
        try:
            lb = KMeans(K, n_init=10, random_state=49).fit_predict(X)
            C += (lb[:, None] == lb[None, :]).astype(float); m += 1
        except Exception: pass
    if m == 0: return ls[0]
    C /= m; D = 1 - C; np.fill_diagonal(D, 0)
    try:
        return AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(D).reshape(-1, 1).astype(float)
    except Exception: return ls[0]
def m_snf(ls, K, t=20):
    """标准 SNF：P_i ← W_i × mean_{j≠i}(P_j) × W_i^T"""
    n = ls[0].shape[0]
    if n < 15: return ls[0]
    k = max(3, min(20, n // 10))
    W = [knn_affinity(X, k) for X in ls]
    P = [w.copy() for w in W]
    for _ in range(t):
        newP = []
        for i in range(len(P)):
            others = np.zeros_like(P[i])
            for j in range(len(P)):
                if j != i: others = others + P[j]
            others /= max(len(P) - 1, 1)
            newP.append(W[i] @ others @ W[i].T)
        P = [(p + p.T) / 2 for p in newP]
    Sf = np.mean(P, axis=0)
    D = 1 - Sf / (Sf.max() + 1e-12); np.fill_diagonal(D, 0)
    try:
        return AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(D).reshape(-1, 1).astype(float)
    except Exception: return ls[0]

METHODS = {
 "1_SNF":            lambda ls, K: m_snf(ls, K),
 "2_NMF(concat)":    lambda ls, K: m_nmf_concat(ls, K),
 "3_KMeans(concat)": lambda ls, K: m_kmeans_concat(ls, K),
 "4_PCA(concat)":    lambda ls, K: m_pca_concat(ls, K),
 "5_谱嵌入(concat)":  lambda ls, K: m_spec_concat(ls, K),
 "6_共联共识":        lambda ls, K: m_consensus(ls, K),
 "7_MCCA-lite":      lambda ls, K: m_mcca_lite(ls, K),
 "8_MOFA-lite":      lambda ls, K: m_mofa_lite(ls, K),
}

def transfer_ari(Xa, Xb, K, seed=49):
    kmA = KMeans(K, n_init=20, random_state=seed).fit(Xa)
    kmB = KMeans(K, n_init=20, random_state=seed).fit(Xb)
    pred = ((Xb[:, None, :] - kmA.cluster_centers_[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    deg = int(len(set(kmB.labels_)) < 2 or len(set(pred)) < 2)
    return float(adjusted_rand_score(kmB.labels_, pred)), deg

def pac(X, K, nboot=NBOOT, seed=49):
    rng = np.random.RandomState(seed); n = X.shape[0]
    if n < 40: return None
    acc = np.zeros((n, n)); cnt = 0
    for _ in range(nboot):
        idx = rng.choice(n, n, replace=True); uniq = np.unique(idx)
        if len(uniq) < K * 3: continue
        try: lb = KMeans(K, n_init=10, random_state=seed).fit_predict(X[uniq])
        except Exception: continue
        pos = {u: i for i, u in enumerate(uniq)}
        for a in range(n):
            for b in range(a + 1, n):
                if a in pos and b in pos:
                    acc[a, b] += (lb[pos[a]] == lb[pos[b]]); acc[b, a] = acc[a, b]
        cnt += 1
    if cnt < 4: return None
    acc /= cnt
    iu = np.triu_indices(n, 1); v = acc[iu]
    return float(np.mean((v > 0.1) & (v < 0.9)))

DOMAINS = {
 "A_CPTAC三队列": {"cohorts": ["ind", "dis", "OV"], "layers": ["mRNA", "CNA", "protein"]},
 "B_EC四层":      {"cohorts": ["TCGA", "ind", "dis"], "layers": ["mRNA", "miRNA", "CNA", "meth"]},
}

res = []
t0 = time.time()
for dname, D in DOMAINS.items():
    cohs, lays = D["cohorts"], D["layers"]
    RAW = {}
    for c in cohs:
        for l in lays:
            M = L(c, l)
            if M is not None: RAW[(c, l)] = M
    print(f"\n{'='*106}\n域 {dname}  队列={cohs}  层={lays}\n{'='*106}", flush=True)
    for c in cohs:
        print(f"  {c:6s} 可用层: {[l for l in lays if (c, l) in RAW]}", flush=True)
    subsets = [s for r in range(1, len(lays)+1) for s in itertools.combinations(lays, r)]
    for sub in subsets:
        for a, b in itertools.combinations(cohs, 2):
            common = [l for l in sub if (a, l) in RAW and (b, l) in RAW]
            if not common: continue
            A_sel, B_sel, ok = [], [], True
            for l in common:
                ga = set(RAW[(a, l)].index); gb = set(RAW[(b, l)].index)
                gk = sorted(ga & gb)
                if len(gk) < 100: ok = False; break
                Xa = zgene(RAW[(a, l)].loc[gk].values.T.astype(float))
                Xb = zgene(RAW[(b, l)].loc[gk].values.T.astype(float))
                ka, kb = topmad_k(Xa, NGENE), topmad_k(Xb, NGENE)
                k = np.array(sorted(set(ka.tolist()) & set(kb.tolist())))
                if len(k) < 50: k = np.arange(min(NGENE, Xa.shape[1]))
                A_sel.append(Xa[:, k]); B_sel.append(Xb[:, k])
            if not ok or not A_sel: continue
            for mname, fn in METHODS.items():
                try:
                    Fa = fn(A_sel, 4); Fb = fn(B_sel, 4)
                except Exception as ex:
                    res.append({"domain": dname, "subset": "+".join(common), "pair": f"{a}×{b}",
                                "method": mname, "ari": None, "err": str(ex)[:70]}); continue
                Fa = zgene(np.nan_to_num(np.asarray(Fa, dtype=float), nan=0.0))
                Fb = zgene(np.nan_to_num(np.asarray(Fb, dtype=float), nan=0.0))
                if Fa.shape[1] != Fb.shape[1] or Fa.shape[0] < 15 or Fb.shape[0] < 15:
                    res.append({"domain": dname, "subset": "+".join(common), "pair": f"{a}×{b}",
                                "method": mname, "ari": None, "err": f"dim {Fa.shape}/{Fb.shape}"}); continue
                aris, degs = {}, {}
                for K in KGRID:
                    try:
                        v, d = transfer_ari(Fa, Fb, K); aris[K] = round(v, 4); degs[K] = d
                    except Exception: aris[K] = None; degs[K] = None
                res.append({"domain": dname, "subset": "+".join(common), "pair": f"{a}×{b}",
                            "method": mname, "ari": aris, "degen": degs,
                            "nA": int(Fa.shape[0]), "nB": int(Fb.shape[0]), "nfeat": int(Fa.shape[1])})
            json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
        print(f"  子集 {'+'.join(sub)} 完成 ({time.time()-t0:.0f}s, 累计 {len(res)} 条)", flush=True)
print(f"\n完成 {len(res)} 条，用时 {time.time()-t0:.0f}s")
print("已落盘", OUT)

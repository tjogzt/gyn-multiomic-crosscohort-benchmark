#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-2：多组学整合基准（修正：按 (队列 × 层子集) 取样本交集）
★ 样本交集本身即「逐层边际价值」的核心量：每加一层，可用样本数如何变化
"""
import os, json, itertools, warnings, time, pickle, gzip
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
OUTA = "/tmp/gyn_retyping/bench_availability.json"

# ---------- 用 CSV 缓存替代 pickle（跨解释器可读） ----------
def _cp(coh, lay): return f"/tmp/gyn_retyping/hcsv/{coh}__{lay}.csv.gz"
def build_csv_cache():
    os.makedirs("/tmp/gyn_retyping/hcsv", exist_ok=True)
    for f in os.listdir(H):
        if not f.endswith(".pkl"): continue
        coh, lay = f[:-4].split("__")
        cp = _cp(coh, lay)
        if os.path.exists(cp): continue
        M = pd.read_pickle(f"{H}/{f}")
        M.to_csv(cp, compression="gzip")
    print("CSV 缓存就绪:", len(os.listdir("/tmp/gyn_retyping/hcsv")))

def L(coh, lay):
    cp = _cp(coh, lay)
    if not os.path.exists(cp): return None
    M = pd.read_csv(cp, index_col=0)
    M.columns = [str(c) for c in M.columns]
    return M

def zgene(X):
    m = np.nanmean(X, axis=0, keepdims=True); s = np.nanstd(X, axis=0, keepdims=True)
    s = np.where((s < 1e-9) | ~np.isfinite(s), 1.0, s)
    return np.nan_to_num((X - m) / s, nan=0.0, posinf=0.0, neginf=0.0)

def topmad_k(X, ng):
    med = np.median(X, axis=0); mad = np.median(np.abs(X - med), axis=0)
    k = np.argsort(-mad)[:ng]; return k[mad[k] > 0]

def knn_affinity(X, k):
    D = euclidean_distances(X)
    nz = D[D > 0]; mu = np.median(nz) if len(nz) else 1.0
    S = np.exp(-D ** 2 / (2 * mu ** 2)); n = S.shape[0]
    K = np.zeros_like(S)
    for i in range(n):
        idx = np.argsort(-S[i])[:k]; K[i, idx] = S[i, idx]
    K = (K + K.T) / 2; rs = K.sum(1, keepdims=True); rs[rs < 1e-12] = 1
    return K / rs

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
    """逐层 KMeans → 每层簇成员指示矩阵（n×K）拼接为特征空间（不返回标签，避免下游伪解）"""
    outs = []
    for X in ls:
        try:
            lb = KMeans(K, n_init=10, random_state=49).fit_predict(X)
            ind = np.zeros((len(lb), K)); ind[np.arange(len(lb)), lb] = 1.0
            outs.append(ind)
        except Exception:
            outs.append(np.zeros((X.shape[0], K)))
    return np.hstack(outs)

def m_snf(ls, K, t=20):
    """标准 SNF → 返回融合网络的谱嵌入（特征空间），不返回标签"""
    n = ls[0].shape[0]
    if n < 15: return ls[0]
    k = max(3, min(20, n // 10))
    W = [knn_affinity(X, k) for X in ls]
    P = [w.copy() for w in W]
    for _ in range(t):
        newP = []
        for i in range(len(P)):
            o = np.zeros_like(P[i])
            for j in range(len(P)):
                if j != i: o = o + P[j]
            o /= max(len(P) - 1, 1)
            newP.append(W[i] @ o @ W[i].T)
        P = [(p + p.T) / 2 for p in newP]
    Sf = np.mean(P, axis=0)
    Sf = (Sf + Sf.T) / 2
    Sf = np.clip(Sf, 0, None)
    d = min(8, n - 2)
    try:
        return SpectralEmbedding(n_components=d, affinity="precomputed",
                                 random_state=49).fit_transform(Sf)
    except Exception:
        return np.hstack(ls)

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

def transfer_ari(Xa, Xb, K, seed=49, ref=None):
    """A 上聚类 → B 样本按最近质心指派 → 与【方法无关的参考划分】比较 ARI
    ref 必须由原始拼接表示导出，避免共识/图方法的结构性自证"""
    kmA = KMeans(K, n_init=20, random_state=seed).fit(Xa)
    pred = ((Xb[:, None, :] - kmA.cluster_centers_[None, :, :]) ** 2).sum(2).argmin(1)
    balA = float(np.bincount(kmA.labels_, minlength=K).max() / len(kmA.labels_))
    balP = float(np.bincount(pred, minlength=K).max() / len(pred))
    if ref is None:
        kmB = KMeans(K, n_init=20, random_state=seed).fit(Xb)
        ref = kmB.labels_
        balB = float(np.bincount(kmB.labels_, minlength=K).max() / len(kmB.labels_))
    else:
        balB = float(np.bincount(ref, minlength=K).max() / len(ref))
    deg = int(min(len(set(kmA.labels_)), len(set(ref)), len(set(pred))) < 2)
    return float(adjusted_rand_score(ref, pred)), deg, balA, balB, balP

DOMAINS = {
 "A_CPTAC三队列": {"cohorts": ["ind", "dis", "OV"], "layers": ["mRNA", "CNA", "protein"]},
 "B_EC四层":      {"cohorts": ["TCGA", "ind", "dis"], "layers": ["mRNA", "miRNA", "CNA", "meth"]},
}
NGENE, KGRID = 1500, [3, 4, 5]

build_csv_cache()
res, avail = [], []
t0 = time.time()
for dname, D in DOMAINS.items():
    cohs, lays = D["cohorts"], D["layers"]
    CACHE = {}
    def g(coh, lay):
        if (coh, lay) not in CACHE: CACHE[(coh, lay)] = L(coh, lay)
        return CACHE[(coh, lay)]
    print(f"\n{'='*106}\n域 {dname}  队列={cohs}  层={lays}\n{'='*106}", flush=True)
    subsets = [s for r in range(1, len(lays)+1) for s in itertools.combinations(lays, r)]
    for sub in subsets:
        # ---- 每队列内：样本交集 ----
        blocks = {}
        for c in cohs:
            Ms = {l: g(c, l) for l in sub}
            if any(v is None for v in Ms.values()): continue
            s = None
            for M in Ms.values():
                s = set(M.columns) if s is None else (s & set(M.columns))
            s = sorted(s)
            if len(s) < 15: continue
            blocks[c] = ({l: M[s] for l, M in Ms.items()}, len(s))
            avail.append({"domain": dname, "subset": "+".join(sub), "cohort": c, "n": len(s)})
        if len(blocks) < 2: continue
        for a, b in itertools.combinations([c for c in cohs if c in blocks], 2):
            A_sel, B_sel, ok = [], [], True
            for l in sub:
                Ma, Mb = blocks[a][0][l], blocks[b][0][l]
                gk = sorted(set(Ma.index) & set(Mb.index))
                if len(gk) < 100: ok = False; break
                Xa = zgene(Ma.loc[gk].values.T.astype(float))
                Xb = zgene(Mb.loc[gk].values.T.astype(float))
                ka, kb = topmad_k(Xa, NGENE), topmad_k(Xb, NGENE)
                k = np.array(sorted(set(ka.tolist()) & set(kb.tolist())))
                if len(k) < 50: k = np.arange(min(NGENE, Xa.shape[1]))
                A_sel.append(Xa[:, k]); B_sel.append(Xb[:, k])
            if not ok or not A_sel: continue
            nA, nB = blocks[a][1], blocks[b][1]
            # 方法无关的参考划分：由 B 的原始拼接表示导出
            REFS = {}
            Bcat = np.hstack(B_sel)
            for K_ in KGRID:
                try: REFS[K_] = KMeans(K_, n_init=20, random_state=49).fit_predict(Bcat)
                except Exception: REFS[K_] = None
            for mname, fn in METHODS.items():
                try:
                    Fa = zgene(np.nan_to_num(np.asarray(fn(A_sel, 4), float), nan=0.0))
                    Fb = zgene(np.nan_to_num(np.asarray(fn(B_sel, 4), float), nan=0.0))
                except Exception as ex:
                    res.append({"domain": dname, "subset": "+".join(sub), "pair": f"{a}×{b}",
                                "method": mname, "ari": None, "err": str(ex)[:70], "nA": nA, "nB": nB}); continue
                if Fa.shape[1] != Fb.shape[1] or min(Fa.shape[0], Fb.shape[0]) < 15:
                    res.append({"domain": dname, "subset": "+".join(sub), "pair": f"{a}×{b}",
                                "method": mname, "ari": None, "err": f"dim {Fa.shape}/{Fb.shape}", "nA": nA, "nB": nB}); continue
                aris, degs, bals = {}, {}, {}
                for K in KGRID:
                    try:
                        v, d, ba, bb, bp = transfer_ari(Fa, Fb, K, ref=REFS[K])
                        aris[K] = round(v, 4); degs[K] = d; bals[K] = [round(ba, 3), round(bb, 3), round(bp, 3)]
                    except Exception:
                        aris[K] = None; degs[K] = None; bals[K] = None
                res.append({"domain": dname, "subset": "+".join(sub), "pair": f"{a}×{b}", "method": mname,
                            "ari": aris, "degen": degs, "balance": bals,
                            "nA": nA, "nB": nB, "nfeat": int(Fa.shape[1])})
            json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
        json.dump(avail, open(OUTA, "w"), indent=1, ensure_ascii=False)
        print(f"  子集 {'+'.join(sub)} 完成 ({time.time()-t0:.0f}s, 累计 {len(res)} 条)", flush=True)
print(f"\n完成 {len(res)} 条，用时 {time.time()-t0:.0f}s")
print("已落盘", OUT, "|", OUTA)

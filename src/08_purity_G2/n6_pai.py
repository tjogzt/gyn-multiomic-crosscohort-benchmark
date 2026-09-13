#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n6：P2 纯度感知整合（PAI）——严格复用基准协议
纯度方向/逐基因纯度相关只在 ind 上学习；dis / TCGA / OV 全部为留出评估队列。
臂：B0 基线 | S1/S2/S3 纯度子空间剔除(k=1,2,3) | G20/G30 纯度无关基因筛选 | R1 纯度残差化(仅 ind×dis)
"""
import os, json, itertools, time
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

HC = "/tmp/gyn_retyping/hcsv"
B_ = "/Volumes/tjogzt4T/data"
OUT = "/tmp/gyn_retyping/pai"
os.makedirs(OUT, exist_ok=True)
NGENE, KGRID = 1500, [3, 4, 5]
SEED = 49

def L(coh, lay):
    p = f"{HC}/{coh}__{lay}.csv.gz"
    if not os.path.exists(p): return None
    M = pd.read_csv(p, index_col=0); M.columns = [str(c) for c in M.columns]
    return M

def zgene(X):
    m = np.nanmean(X, 0, keepdims=True); s = np.nanstd(X, 0, keepdims=True)
    s = np.where((s < 1e-9) | ~np.isfinite(s), 1.0, s)
    return np.nan_to_num((X - m) / s, nan=0.0, posinf=0.0, neginf=0.0)

def topmad_k(X, ng):
    med = np.median(X, 0); mad = np.median(np.abs(X - med), 0)
    k = np.argsort(-mad)[:ng]; return k[mad[k] > 0]

def transfer_ari(Xa, Xb, K, ref, seed=SEED):
    kmA = KMeans(K, n_init=20, random_state=seed).fit(Xa)
    pred = ((Xb[:, None, :] - kmA.cluster_centers_[None, :, :]) ** 2).sum(2).argmin(1)
    deg = int(min(len(set(kmA.labels_)), len(set(ref)), len(set(pred))) < 2)
    return float(adjusted_rand_score(ref, pred)), deg

# ---------- 在 ind 上学习纯度信息 ----------
print("在 ind 上学习纯度方向与逐基因纯度相关 …")
ind_meta = pd.read_excel(f"{B_}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
_t = pd.DataFrame({"id": ind_meta["Case_id"].astype(str),
                   "p": pd.to_numeric(ind_meta["ABSOLUTE_tumor_purity"], errors="coerce")}).dropna()
_t = _t[_t["p"] > 0].drop_duplicates("id").set_index("id")["p"]
PUR = _t
print(f"  学习样本 {len(PUR)} 例 | 纯度均值 {PUR.mean():.4f}")

LEARN = {}          # layer -> {"genes": [...], "r": array, "dirs": [u1,u2,u3], "intercept":…, "slope":…}
for lay in ["mRNA", "miRNA", "CNA", "meth", "protein"]:
    M = L("ind", lay)
    if M is None: continue
    s = [c for c in M.columns if c in PUR.index]
    if len(s) < 40: continue
    X = M[s].T.astype(float)                       # 样本 × 基因
    X = X.loc[:, X.notna().mean() > 0.9]
    X = X.fillna(X.median())
    Z = zgene(X.values)
    y = PUR.reindex([str(i) for i in X.index]).values.astype(float)
    # 向量化皮尔逊相关（替代逐基因循环）
    Zc = Z - Z.mean(0, keepdims=True); yc = y - y.mean()
    den = np.sqrt((Zc ** 2).sum(0)) * np.sqrt((yc ** 2).sum())
    den = np.where(den < 1e-12, 1.0, den)
    r = np.nan_to_num((Zc * yc[:, None]).sum(0) / den)
    # 逐基因 OLS 斜率（残差化用）
    yc = y - y.mean()
    slope = (Z * yc[:, None]).sum(0) / max((yc ** 2).sum(), 1e-9)
    # 纯度方向：用【数据自身的与纯度相关 PC 子空间】
    # 注：基因空间中与单一纯度变量对齐的方向只有 1 维（deflation 后协方差恒为 0），
    #     故必须改用样本空间的“纯度相关 PC”，才能得到真正的 k 维可移除子空间。
    from sklearn.decomposition import PCA
    npc = min(20, Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=npc, random_state=49).fit(Z)
    Ts = pca.transform(Z)
    rpc = np.array([abs(np.corrcoef(Ts[:, i], y)[0, 1]) if np.std(Ts[:, i]) > 0 else 0.0
                    for i in range(npc)])
    rpc = np.nan_to_num(rpc)
    order = np.argsort(-rpc)                      # 按 |r| 降序的 PC 顺序
    dirs = [pca.components_[i] for i in order]    # 每个都是基因空间单位向量
    LEARN[lay] = {"genes": list(X.columns), "r": r, "dirs": dirs, "slope": slope,
                  "pc_r_sorted": rpc[order].round(4).tolist()}
    print(f"  {lay:8s} 基因 {len(X.columns):6d} | |r|>0.3 占比 {np.mean(np.abs(r)>0.3)*100:5.1f}% "
          f"| 纯度相关 PC 的 |r| 前3 = {rpc[order][:3].round(3).tolist()}")

json.dump({k: {"n_gene": len(v["genes"]), "frac_r_gt_0.3": float(np.mean(np.abs(v["r"]) > 0.3)),
               "n_dir": len(v["dirs"])} for k, v in LEARN.items()},
          open(f"{OUT}/learn_summary.json", "w"), ensure_ascii=False, indent=1)

# ---------- 变换 ----------
def take(meta, genes_sel):
    """把 ind 学到的量对齐到被选中的基因上"""
    idx = {g: i for i, g in enumerate(meta["genes"])}
    pos = np.array([idx[g] for g in genes_sel if g in idx])
    keep = np.array([i for i, g in enumerate(genes_sel) if g in idx])
    return pos, keep

def apply_arm(X, sel, lay, arm, purity=None):
    """X: 样本 × 选中基因（已 z-score）；sel: 对应的基因名；purity: 该队列的逐样本纯度或 None"""
    meta = LEARN.get(lay)
    if meta is None: return X, 0
    pos, keep = take(meta, sel)
    if len(keep) < 20: return X, 0
    if arm.startswith("S"):
        k = int(arm[1:])
        Y = X.copy()
        for u in meta["dirs"][:k]:
            uu = u[pos]; n = np.linalg.norm(uu)
            if n < 1e-9: continue
            uu = uu / n
            Y[:, keep] = Y[:, keep] - (Y[:, keep] @ uu)[:, None] * uu[None, :]
        return zgene(Y), len(keep)
    if arm.startswith("G"):
        g = float(arm[1:]) / 100.0
        m = np.abs(meta["r"][pos]) < g          # 保留：纯度无关
        Y = X.copy()
        Y[:, keep[~m]] = 0.0                    # 置零 = 从欧氏距离中移除
        return zgene(Y), int(m.sum())
    if arm == "R1":
        if purity is None: return None, 0        # 该队列无纯度 → 不适用
        beta = meta["slope"][pos]
        pc = purity - np.nanmean(purity)
        Y = X.copy()
        Y[:, keep] = Y[:, keep] - np.outer(pc, beta)
        return zgene(Y), len(keep)
    return X, 0

# ---------- 主循环 ----------
DOMAINS = {
 "A_CPTAC三队列": {"cohorts": ["ind", "dis", "OV"], "layers": ["mRNA", "CNA", "protein"]},
 "B_EC四层":      {"cohorts": ["TCGA", "ind", "dis"], "layers": ["mRNA", "miRNA", "CNA", "meth"]},
}
ARMS = (os.environ.get("PAI_ARMS") or "B0,S1,S2,S3,G20,G30,R1").split(",")

# 逐样本纯度（用于 R1；仅 ind/dis 可得）
dc = pd.read_csv(f"{B_}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                 sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
_dd = pd.DataFrame({"id": dc["Proteomics_Participant_ID"].astype(str),
                    "p": pd.to_numeric(dc["Purity_Cancer"], errors="coerce"),
                    "tn": dc["Proteomics_Tumor_Normal"].astype(str)})
PUR_BY_COHORT = {"ind": PUR,
                 "dis": _dd[(_dd["tn"] == "Tumor") & _dd["p"].notna()].drop_duplicates("id").set_index("id")["p"]}
print(f"逐样本纯度可得队列: {[(k, len(v)) for k, v in PUR_BY_COHORT.items()]}")

CACHE = {}
def g(coh, lay):
    if (coh, lay) not in CACHE: CACHE[(coh, lay)] = L(coh, lay)
    return CACHE[(coh, lay)]

res, t0 = [], time.time()
for dname, D in DOMAINS.items():
    cohs, lays = D["cohorts"], D["layers"]
    print(f"\n{'='*100}\n域 {dname}\n{'='*100}", flush=True)
    subsets = [s for r in range(1, len(lays) + 1) for s in itertools.combinations(lays, r)]
    for sub in subsets:
        blocks = {}
        for c in cohs:
            Ms = {l: g(c, l) for l in sub}
            if any(v is None for v in Ms.values()): continue
            s = None
            for M in Ms.values(): s = set(M.columns) if s is None else (s & set(M.columns))
            s = sorted(s)
            if len(s) < 15: continue
            blocks[c] = ({l: M[s] for l, M in Ms.items()}, len(s))
        if len(blocks) < 2: continue
        for a, b in itertools.combinations([c for c in cohs if c in blocks], 2):
            araw, braw, ok = [], [], True
            for l in sub:
                Ma, Mb = blocks[a][0][l], blocks[b][0][l]
                gk = sorted(set(Ma.index) & set(Mb.index))
                if len(gk) < 100: ok = False; break
                Xa = zgene(Ma.loc[gk].values.T.astype(float))
                Xb = zgene(Mb.loc[gk].values.T.astype(float))
                ka, kb = topmad_k(Xa, NGENE), topmad_k(Xb, NGENE)
                k = np.array(sorted(set(ka.tolist()) & set(kb.tolist())))
                if len(k) < 50: k = np.arange(min(NGENE, Xa.shape[1]))
                # 关键：必须与基准一致，只用 top-1500 MAD 交集的子空间
                araw.append((Xa[:, k], [gk[i] for i in k], l))
                braw.append((Xb[:, k], [gk[i] for i in k], l))
            if not ok or not araw: continue
            Bcat0 = np.hstack([x[0] for x in braw])
            REFS = {K_: KMeans(K_, n_init=20, random_state=SEED).fit_predict(Bcat0) for K_ in KGRID}
            for arm in ARMS:
                A_sel, B_sel, kept, skip = [], [], [], False
                for (Xa, sel, l), (Xb, _, _) in zip(araw, braw):
                    pa = None if a not in PUR_BY_COHORT else PUR_BY_COHORT[a].reindex([str(i) for i in blocks[a][0][l].columns]).values
                    pb = None if b not in PUR_BY_COHORT else PUR_BY_COHORT[b].reindex([str(i) for i in blocks[b][0][l].columns]).values
                    Ya, ka = apply_arm(Xa, sel, l, arm, pa)
                    Yb, _ = apply_arm(Xb, sel, l, arm, pb)
                    if Ya is None or Yb is None: skip = True; break
                    A_sel.append(Ya); B_sel.append(Yb); kept.append(ka)
                if skip:
                    res.append({"domain": dname, "subset": "+".join(sub), "pair": f"{a}×{b}", "arm": arm,
                                "ari": None, "err": "该队列无纯度数据（R1 不适用）",
                                "nA": blocks[a][1], "nB": blocks[b][1], "nfeat": None, "n_purity_indep": None})
                    json.dump(res, open(f"{OUT}/pai_matrix.json", "w"), indent=1, ensure_ascii=False)
                    continue
                Fa, Fb = zgene(np.hstack(A_sel)), zgene(np.hstack(B_sel))
                aris, degs = {}, {}
                for K_ in KGRID:
                    v, d = transfer_ari(Fa, Fb, K_, REFS[K_])
                    aris[K_] = round(v, 4); degs[K_] = d
                res.append({"domain": dname, "subset": "+".join(sub), "pair": f"{a}×{b}", "arm": arm,
                            "ari": aris, "degen": degs, "nA": blocks[a][1], "nB": blocks[b][1],
                            "nfeat": int(Fa.shape[1]), "n_purity_indep": int(np.sum(kept))})
            json.dump(res, open(f"{OUT}/pai_matrix.json", "w"), indent=1, ensure_ascii=False)
        print(f"  子集 {'+'.join(sub)} 完成 ({time.time()-t0:.0f}s, 累计 {len(res)})", flush=True)
print(f"\n完成 {len(res)} 条，用时 {time.time()-t0:.0f}s")

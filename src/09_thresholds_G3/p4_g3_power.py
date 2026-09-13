#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p4：G3 门槛反推——NSMP 亚结构的可检出性
① 队列内 split-half ARI（可复现性天花板）
② 随机划分 null（AR 度量的噪声尺度）
③ 跨队列迁移 ARI（G3 主统计量，实测值）
"""
import json, os, itertools
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

B = "/Volumes/tjogzt4T/data"; T = "/tmp/gyn_retyping"; H = f"{T}/hcsv"
LAYS = ["mRNA", "miRNA", "CNA", "meth"]
KS = [2, 3, 4]
NGENE = 1000
rng = np.random.RandomState(49)

def L(coh, lay):
    p = f"{H}/{coh}__{lay}.csv.gz"
    if not os.path.exists(p): return None
    M = pd.read_csv(p, index_col=0); M.columns = [str(c) for c in M.columns]
    return M

# ---------- NSMP 三队列 ----------
ns = json.load(open(f"{T}/ucec_core_nsmp.json"))
tc_nsmp = sorted({s[:12] for s in ns["nsmp_core"]})
x = pd.read_excel(f"{B}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
v1_ids = set(x.loc[x["Genomic_subtype"] == "CNV_L", "Case_id"].astype(str))
dc = pd.read_csv(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                 sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
v2_ids = set(dc.loc[dc["CNV_class"] == "CNV_LOW", "Proteomics_Participant_ID"].astype(str))
print(f"NSMP 名单: TCGA {len(tc_nsmp)} | V1 {len(v1_ids)} | V2 {len(v2_ids)}")

def z(X):
    m = np.nanmean(X, 0, keepdims=True); s = np.nanstd(X, 0, keepdims=True)
    s = np.where((s < 1e-9) | ~np.isfinite(s), 1.0, s)
    return np.nan_to_num((X - m) / s, nan=0.0, posinf=0.0, neginf=0.0)

def build(coh, ids, match12=False):
    """返回 (逐层矩阵列表, 样本名)。只用四层都有数据的样本。"""
    idkey = {i[:12] for i in ids} if match12 else set(ids)
    per_layer, keep = [], None
    for lay in LAYS:
        M = L(coh, lay)
        if M is None:
            print(f"    [{coh}/{lay}] 层缺失"); return None, None
        cmap = {str(c)[:12] if match12 else str(c): str(c) for c in M.columns}
        avail = set(cmap) & idkey
        print(f"    [{coh}/{lay}] 列 {len(cmap)} | 交集 {len(avail)}")
        keep = avail if keep is None else (keep & avail)
        per_layer.append((lay, M, cmap))
    keep = sorted(keep)
    print(f"    [{coh}] 四层交集 {len(keep)}")
    if len(keep) < 20:
        print(f"    [{coh}] 样本不足 → 返回空"); return None, None
    blocks = []
    for lay, M, cmap in per_layer:
        sub = [cmap[k] for k in keep]
        blocks.append(z(M[sub].T.values.astype(float)))
    return blocks, keep

def cat_sel(bl_a, bl_b):
    """逐层共同基因 → z → top-N MAD 交集 → 拼接 → z"""
    A, Bb = [], []
    for Xa, Xb in zip(bl_a, bl_b):
        g = min(Xa.shape[1], Xb.shape[1])
        Xa, Xb = Xa[:, :g], Xb[:, :g]
        ka = np.argsort(-np.median(np.abs(Xa - np.median(Xa, 0)), 0))[:NGENE]
        kb = np.argsort(-np.median(np.abs(Xb - np.median(Xb, 0)), 0))[:NGENE]
        k = np.array(sorted(set(ka.tolist()) & set(kb.tolist())))
        A.append(Xa[:, k]); Bb.append(Xb[:, k])
    return z(np.hstack(A)), z(np.hstack(Bb))

TC = {}
for name, code, ids, m12 in [("TCGA", "TCGA", tc_nsmp, True), ("V1", "ind", v1_ids, False), ("V2", "dis", v2_ids, False)]:
    bl, keep = build(code, ids, m12)
    TC[name] = (bl, keep)
    print(f"  {name} (层文件 {code}): 四层交集样本 {len(keep) if keep else 0}")

# ---------- ① split-half 天花板 ----------
print("\n=== ① 队列内 split-half ARI（可复现性天花板）===")
SH = {}
for name in ["TCGA", "V1", "V2"]:
    bl, keep = TC[name]
    if bl is None or len(keep) < 30: print(f"  {name}: 样本不足"); continue
    vals = {k: [] for k in KS}
    for rep in range(200):
        idx = rng.permutation(len(keep))
        h1, h2 = idx[:len(idx)//2], idx[len(idx)//2:2*(len(idx)//2)]
        A = z(np.hstack([X[h1] for X in bl])); Bx = z(np.hstack([X[h2] for X in bl]))
        for k in KS:
            p1 = KMeans(k, n_init=10, random_state=rep).fit_predict(A)
            p2 = KMeans(k, n_init=10, random_state=rep+1000).fit_predict(Bx)
            vals[k].append(adjusted_rand_score(p1, p2))
    SH[name] = {str(k): float(np.mean(v)) for k, v in vals.items()}
    print(f"  {name} (n={len(keep)}): " + " | ".join(f"K={k} ARI {np.mean(v):.4f}±{np.std(v):.4f}" for k, v in vals.items()))

# ---------- ② 随机划分 null ----------
print("\n=== ② 随机划分 null（AR 度量的噪声尺度）===")
NULL = {}
for n in [41, 67, 71]:
    for k in KS:
        v = [adjusted_rand_score(rng.permutation(k)[rng.randint(0, k, n)],
                                 rng.permutation(k)[rng.randint(0, k, n)]) for _ in range(3000)]
        v = np.array(v)
        NULL[f"n{n}_K{k}"] = {"mean": float(v.mean()), "sd": float(v.std()),
                              "q95": float(np.percentile(v, 95)), "q99": float(np.percentile(v, 99))}
print("  （仅列 n=71 与 n=41 的关键值）")
for key in ["n71_K2","n71_K3","n71_K4","n41_K2","n41_K3","n41_K4"]:
    d = NULL[key]; print(f"  {key}: 均值 {d['mean']:+.4f} SD {d['sd']:.4f} Q95 {d['q95']:+.4f} Q99 {d['q99']:+.4f}")

# ---------- ③ 跨队列迁移 ARI（实测） ----------
print("\n=== ③ 跨队列迁移 ARI（G3 主统计量，实测）===")
CROSS = {}
for a, b in [("TCGA", "V1"), ("TCGA", "V2"), ("V1", "V2")]:
    if TC[a][0] is None or TC[b][0] is None: continue
    Fa, Fb = cat_sel(TC[a][0], TC[b][0])
    res = {}
    for k in KS:
        kmA = KMeans(k, n_init=20, random_state=49).fit(Fa)
        pred = ((Fb[:, None, :] - kmA.cluster_centers_[None]) ** 2).sum(2).argmin(1)
        refB = KMeans(k, n_init=20, random_state=49).fit_predict(Fb)
        res[str(k)] = round(float(adjusted_rand_score(refB, pred)), 4)
    CROSS[f"{a}→{b}"] = res
    print(f"  {a}→{b} (n={Fa.shape[0]}→{Fb.shape[0]}, d={Fa.shape[1]}): " +
          " | ".join(f"K={k} {v:+.4f}" for k, v in res.items()))

json.dump({"split_half": SH, "null": NULL, "cross": CROSS,
           "n": {k: (len(TC[k][1]) if TC[k][1] else 0) for k in TC}},
          open(f"{T}/g3_power.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 g3_power.json")

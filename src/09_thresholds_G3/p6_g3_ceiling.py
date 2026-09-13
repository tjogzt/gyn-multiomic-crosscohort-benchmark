#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p6：G3 冻结所需的两个补充判据
① 全样本聚类的 bootstrap 稳定性（正确的可复现性天花板）
② 跨队列运行的退化诊断（最大簇占比）——查 K=2 ARI=1.0 是否退化
"""
import json, os
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

T = "/tmp/gyn_retyping"; H = f"{T}/hcsv"
LAYS = ["mRNA", "miRNA", "CNA", "meth"]; KS = [2, 3, 4]; NGENE = 1000
rng = np.random.RandomState(49)

def L(coh, lay):
    p = f"{H}/{coh}__{lay}.csv.gz"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None

def z(X):
    m = np.nanmean(X, 0, keepdims=True); s = np.nanstd(X, 0, keepdims=True)
    s = np.where((s < 1e-9) | ~np.isfinite(s), 1.0, s)
    return np.nan_to_num((X - m) / s, nan=0.0, posinf=0.0, neginf=0.0)

G3 = json.load(open(f"{T}/g3_power.json"))
ns = json.load(open(f"{T}/ucec_core_nsmp.json"))
tc = sorted({s[:12] for s in ns["nsmp_core"]})

# 重建三个队列的块（与 p4 同逻辑，此处直接内联）
import importlib.util
def build_blocks(code, ids, match12):
    idkey = {i[:12] for i in ids} if match12 else set(ids)
    per, keep = [], None
    for lay in LAYS:
        M = L(code, lay)
        cmap = {str(c)[:12] if match12 else str(c): str(c) for c in M.columns}
        avail = set(cmap) & idkey
        keep = avail if keep is None else (keep & avail)
        per.append((M, cmap))
    keep = sorted(keep); blocks = []
    for M, cmap in per:
        blocks.append(z(M[[cmap[k] for k in keep]].T.values.astype(float)))
    return blocks, keep

B = "/Volumes/tjogzt4T/data"
x = pd.read_excel(f"{B}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
v1 = set(x.loc[x["Genomic_subtype"] == "CNV_L", "Case_id"].astype(str))
dc = pd.read_csv(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                 sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
v2 = set(dc.loc[dc["CNV_class"] == "CNV_LOW", "Proteomics_Participant_ID"].astype(str))
SETS = {"TCGA": build_blocks("TCGA", tc, True), "V1": build_blocks("ind", v1, False), "V2": build_blocks("dis", v2, False)}

def cat(bl_a, bl_b):
    A, Bb = [], []
    for Xa, Xb in zip(bl_a, bl_b):
        g = min(Xa.shape[1], Xb.shape[1]); Xa, Xb = Xa[:, :g], Xb[:, :g]
        ka = np.argsort(-np.median(np.abs(Xa - np.median(Xa, 0)), 0))[:NGENE]
        kb = np.argsort(-np.median(np.abs(Xb - np.median(Xb, 0)), 0))[:NGENE]
        k = np.array(sorted(set(ka.tolist()) & set(kb.tolist())))
        A.append(Xa[:, k]); Bb.append(Xb[:, k])
    return z(np.hstack(A)), z(np.hstack(Bb))

# ---------- ① bootstrap 稳定性（全样本聚类的天花板）----------
print("=== ① 全样本聚类的 bootstrap 稳定性（正确天花板）===")
BOOT = {}
for name, (bl, keep) in SETS.items():
    F = z(np.hstack(bl)); n = F.shape[0]
    row = {}
    for k in KS:
        base = KMeans(k, n_init=20, random_state=49).fit_predict(F)
        mf = float(np.bincount(base, minlength=k).max() / len(base))
        vals = []
        for rep in range(200):
            idx = rng.randint(0, n, n)
            if len(set(idx)) < k + 2: continue
            p = KMeans(k, n_init=10, random_state=rep).fit_predict(F[idx])
            vals.append(adjusted_rand_score(base[idx], p))
        row[str(k)] = {"boot_ari": float(np.mean(vals)), "boot_sd": float(np.std(vals)),
                       "maxfrac_base": round(mf, 3)}
        print(f"  {name} (n={n}) K={k}: bootstrap ARI {np.mean(vals):+.4f}±{np.std(vals):.4f} | 全样本最大簇占比 {mf:.3f}")
    BOOT[name] = row

# ---------- ② 跨队列退化诊断 ----------
print("\n=== ② 跨队列运行退化诊断（最大簇占比）===")
DEG = {}
for a, b in [("TCGA", "V1"), ("TCGA", "V2"), ("V1", "V2")]:
    Fa, Fb = cat(SETS[a][0], SETS[b][0])
    row = {}
    for k in KS:
        kmA = KMeans(k, n_init=20, random_state=49).fit(Fa)
        pred = ((Fb[:, None, :] - kmA.cluster_centers_[None]) ** 2).sum(2).argmin(1)
        refB = KMeans(k, n_init=20, random_state=49).fit_predict(Fb)
        ari = float(adjusted_rand_score(refB, pred))
        mA = float(np.bincount(kmA.labels_, minlength=k).max() / len(kmA.labels_))
        mB = float(np.bincount(refB, minlength=k).max() / len(refB))
        mP = float(np.bincount(pred, minlength=k).max() / len(pred))
        row[str(k)] = {"ari": round(ari, 4), "maxfrac_A": round(mA, 3), "maxfrac_refB": round(mB, 3),
                       "maxfrac_pred": round(mP, 3), "degen": int(max(mA, mB, mP) > 0.8)}
        print(f"  {a}→{b} K={k}: ARI {ari:+.4f} | 最大簇 A {mA:.3f} refB {mB:.3f} pred {mP:.3f}"
              + ("  ⚠️退化" if row[str(k)]["degen"] else ""))
    DEG[f"{a}→{b}"] = row

out = dict(G3)
out["bootstrap_ceiling"] = BOOT
out["cross_degeneracy"] = DEG
json.dump(out, open(f"{T}/g3_power.json", "w"), ensure_ascii=False, indent=1)
print("\n已更新 g3_power.json")

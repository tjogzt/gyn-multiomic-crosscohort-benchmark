#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-10：Python 侧与 R 侧方法合并排名 + 结论一致性检验"""
import json, sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

BASE = "/tmp/gyn_retyping"
RPATH = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/bench_r_matrix.csv"
K = "4"

# ---------- Python 侧 ----------
P = pd.DataFrame(json.load(open(f"{BASE}/bench_matrix.json")))
P["ari_k"] = P["ari"].map(lambda d: d.get(K) if isinstance(d, dict) else None)
P["dom"] = P["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
print("Python:", len(P), "条 | 方法:", sorted(P["method"].unique()))

# ---------- R 侧 ----------
if not os.path.exists(RPATH):
    print("R 侧结果未落盘"); raise SystemExit
R = pd.read_csv(RPATH)
R["ari_k"] = R[f"ari_{K}"]
R["dom"] = R["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
print("R     :", len(R), "条 | 方法:", sorted(R["method"].unique()))
ne = int(R["err"].notna().sum()) if "err" in R.columns else 0
print("R 侧错误:", ne, "/", len(R))

norm = lambda s: "+".join(sorted(str(s).replace("+", " ").split()))
for D in (P, R):
    D["sub"] = D["subset"].map(norm)
    D["pairn"] = D["pair"].astype(str).str.replace("×", "x").str.replace("X", "x").str.strip()

# ---------- 1) R 侧方法排名 ----------
print(f"\n=== 1) R 侧方法排名（迁移 ARI，K={K}）===")
rk = R.dropna(subset=["ari_k"]).groupby("method")["ari_k"].agg(["mean", "count"]).sort_values("mean", ascending=False)
print(rk.round(4).to_string())

# ---------- 2) R 侧层数效应 ----------
print("\n=== 2) R 侧层数效应（域 × 层数）===")
R["nlay"] = R["sub"].map(lambda s: len(s.split("+")))
le = R.dropna(subset=["ari_k"]).groupby(["dom", "nlay"])["ari_k"].agg(["mean", "count"]).round(4)
print(le.to_string())

# ---------- 3) Python 侧层数效应（同口径复算）----------
print("\n=== 3) Python 侧层数效应（同口径）===")
P["nlay"] = P["sub"].map(lambda s: len(s.split("+")))
le2 = P.dropna(subset=["ari_k"]).groupby(["dom", "nlay"])["ari_k"].agg(["mean", "count"]).round(4)
print(le2.to_string())

# ---------- 4) 跨侧一致性（同格子）----------
print("\n=== 4) 同 (域×层子集×队列对) 格子上两侧是否一致 ===")
key = ["dom", "sub", "pairn"]
Pa = P.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("PY")
Ra = R.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("R")
cmp_ = pd.concat([Pa, Ra], axis=1).dropna()
res = {"n_grid": int(len(cmp_))}
if len(cmp_) >= 5:
    rho, pv = spearmanr(cmp_["PY"], cmp_["R"])
    res.update(spearman=float(rho), p=float(pv), py_mean=float(cmp_["PY"].mean()),
               r_mean=float(cmp_["R"].mean()),
               median_absdiff=float(np.median(np.abs(cmp_["PY"] - cmp_["R"]))))
    print(f"共同格 {len(cmp_)} 个")
    print(f"  Python 均值 {cmp_['PY'].mean():.4f} | R 均值 {cmp_['R'].mean():.4f}")
    print(f"  格子级 Spearman = {rho:.4f}  (p={pv:.3g})")
    print(f"  绝对差中位 = {np.median(np.abs(cmp_['PY']-cmp_['R'])):.4f}")
else:
    print("可比较格子不足:", len(cmp_))

# ---------- 5) 合并总排名 ----------
print("\n=== 5) 合并总排名（Python + R 全部方法，记录加权）===")
allm = pd.concat([P[["method", "ari_k"]].assign(side="PY"), R[["method", "ari_k"]].assign(side="R")])
tot = allm.dropna(subset=["ari_k"]).groupby(["side", "method"])["ari_k"].agg(["mean", "count"]).sort_values("mean", ascending=False)
print(tot.round(4).to_string())

# ---------- 6) 关键结论是否被 R 侧复现 ----------
print("\n=== 6) 关键结论复查 ===")
# (a) 层数负边际价值
for dom in ["A", "B"]:
    a = le.loc[(dom, 1), "mean"] if (dom, 1) in le.index else np.nan
    b = le.loc[(dom, max(le.loc[dom].index.get_level_values(0))), "mean"] if dom in le.index.get_level_values(0) else np.nan
    print(f"  域{dom}: R 侧 1层 {a:.4f} → 最多层 {b:.4f}")
# (b) 最简方法是否领先
print("  R 侧最优方法:", rk.index[0], f"({rk['mean'].iloc[0]:.4f})")
print(f"  {'✓' if rk['mean'].iloc[0] < rk['mean'].iloc[-1] else '✗'} 最优/最差差距 = {rk['mean'].iloc[0]-rk['mean'].iloc[-1]:.4f}")

out = {"r_ranking": rk["mean"].round(4).to_dict(), "r_counts": rk["count"].to_dict(),
       "r_layer_effect": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in le["mean"].items()},
       "py_layer_effect": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in le2["mean"].items()},
       "r_errors": ne, "r_records": int(len(R)), "py_records": int(len(P)),
       "cross_side": res, "r_nlay_max_mean": None}
json.dump(out, open(f"{BASE}/bench_merged.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 bench_merged.json")

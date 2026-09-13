#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n7：P2 结果聚合 + G2 判定
G2 冻结判据：基线 B0 = 0.4027（单一方法 KMeans(concat)）；阈值 +46.6%；绝对目标 0.5905
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

OUT = "/tmp/gyn_retyping/pai"
G2 = json.load(open("/tmp/gyn_retyping/g2_refined.json"))
BASE_V = G2["baseline_value"]; SD_R = G2["sd_relevant"]
print(f"G2 冻结参数: 基线 {G2['baseline']} = {BASE_V:.4f} | 阈值 +46.6% → 绝对目标 {BASE_V*1.466:.4f}")

D = json.load(open(f"{OUT}/pai_matrix.json"))
df = pd.DataFrame(D)
df = df[df["ari"].notna()].copy()
for k in ["3", "4", "5"]:
    df[f"ari{k}"] = df["ari"].map(lambda d: d.get(k) if isinstance(d, dict) else None)
    df[f"deg{k}"] = df["degen"].map(lambda d: d.get(k) if isinstance(d, dict) else None)
df = df[df["deg4"].fillna(1) == 0]
df["rec"] = df["domain"] + "|" + df["subset"] + "|" + df["pair"]
print(f"\n有效记录 {len(df)} | 臂: {sorted(df['arm'].unique())}")
print(f"单元数（域×层子集×队列对）: {df['rec'].nunique()}")

K = "ari4"
tab = df.groupby("arm")[K].agg(["mean", "count"]).sort_values("mean", ascending=False)
print("\n=== 各臂 K=4 迁移 ARI ===")
print(tab.round(4).to_string())

ARMN = {"B0": "B0 基线（无纯度处理）", "S1": "S1 剔除纯度PC k=1", "S2": "S2 剔除纯度PC k=2",
        "S3": "S3 剔除纯度PC k=3", "G20": "G20 纯度无关基因 γ=0.2", "G30": "G30 纯度无关基因 γ=0.3"}

print("\n=== 与基线 B0 的配对比较（同记录）===")
rows = []
for arm in sorted(df["arm"].unique()):
    if arm == "B0": continue
    a = df[df["arm"] == arm].set_index("rec")[K]
    b = df[df["arm"] == "B0"].set_index("rec")[K]
    j = pd.concat([a.rename("arm"), b.rename("base")], axis=1).dropna()
    if len(j) < 5:
        print(f"  {ARMN.get(arm,arm):28s} 可比记录不足 ({len(j)})"); continue
    d = j["arm"] - j["base"]
    t, p = stats.ttest_rel(j["arm"], j["base"])
    rel = d.mean() / BASE_V
    rows.append({"arm": ARMN.get(arm, arm), "n": len(j), "mean_arm": round(j["arm"].mean(), 4),
                 "mean_delta": round(d.mean(), 4), "rel_vs_base": round(rel, 4),
                 "p": float(p), "ci_lo": round(d.mean() - 1.96*d.std(ddof=1)/np.sqrt(len(j)), 4),
                 "ci_hi": round(d.mean() + 1.96*d.std(ddof=1)/np.sqrt(len(j)), 4),
                 "win_rate": round(float((d > 0).mean()), 4)})
    print(f"  {ARMN.get(arm,arm):28s} n={len(j):3d}  ΔARI={d.mean():+.4f}  相对 {rel*100:+6.1f}%  "
          f"95%CI[{d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(j)):+.4f},{d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(j)):+.4f}]  "
          f"胜率 {(d>0).mean()*100:4.1f}%  配对p={p:.3g}")
T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/pai_compare.csv", index=False)

# ---------- 分域 ----------
print("\n=== 分域（K=4 均值）===")
pv = df.pivot_table(index=["domain", "arm"], values=K, aggfunc="mean").round(4)
print(pv.to_string())

# ---------- 分层（单层子集，看蛋白层是否受益更多）----------
df["nlay"] = df["subset"].map(lambda s: len(str(s).split("+")))
print("\n=== 按层数（K=4 均值）===")
print(df.pivot_table(index=["nlay", "arm"], values=K, aggfunc="mean").round(4).to_string())

# ---------- G2 判定 ----------
print("\n" + "=" * 88)
print("G2 判定（冻结判据）")
print("=" * 88)
best = tab.index[0]; bestv = tab.loc[best, "mean"]
if best == "B0":
    print(f"  最佳臂即基线本身（{bestv:.4f}）→ 纯度感知**未带来任何提升**")
    print("  → G2 = ❌ 未通过（但见下方不可判核查）")
else:
    need = BASE_V * 1.466
    print(f"  最佳臂 {ARMN.get(best,best)} = {bestv:.4f} | 基线 {BASE_V:.4f} | 绝对目标 {need:.4f}")
    print(f"  → {'✅ 通过' if bestv >= need else '❌ 未通过'}（相对提升 {(bestv/BASE_V-1)*100:+.1f}%，阈值 +46.6%）")
# 不可判核查：能否排除 +46.6%
if len(T):
    r = T.iloc[0]
    need_abs = 0.466 * BASE_V
    print(f"\n  不可判核查（预注册条款）：观测相对提升上界 = 95%CI 上界 {r['ci_hi']/BASE_V*100:+.1f}%")
    print(f"    能否排除 +46.6%？{'能（未过门确定）' if r['ci_hi'] < need_abs else '不能 → 记「不可判（under-powered）」'}")
json.dump({"table": tab["mean"].round(4).to_dict(), "compare": rows,
           "g2_base": BASE_V, "g2_target": BASE_V * 1.466,
           "n_records": int(len(df)), "n_units": int(df["rec"].nunique())},
          open(f"{OUT}/pai_results.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 pai_results.json")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m1：用效应量分布反推 G2 阈值
三个噪声来源 → 最小可检测差异（MDD）→ 相对阈值
 ① 留出队列选择：同一 (域,层子集,方法) 上跨队列对的 ARI 波动
 ② K 选择：同一记录上跨 K=3/4/5 的 ARI 波动
 ③ 方法比较（配对差）：同一记录上两方法 ΔARI 的分布
再加 ④ 选择偏倚：基线是"筛出来的最优"，须扣除 max-of-k 的乐观偏倚
"""
import json
import numpy as np, pandas as pd
from scipy import stats

BASE = "/tmp/gyn_retyping"
rng = np.random.RandomState(49)

def load_py():
    df = pd.DataFrame(json.load(open(f"{BASE}/bench_matrix.json")))
    for k in ["3", "4", "5"]:
        df[f"ari{k}"] = df["ari"].map(lambda d: d.get(k) if isinstance(d, dict) else None)
    df["degen4"] = df["degen"].map(lambda d: d.get("4") if isinstance(d, dict) else None)
    return df

def load_r():
    df = pd.read_csv(f"{BASE}/bench_r_matrix.csv")
    return df

P = load_py(); R = load_r()
print(f"Python 记录 {len(P)} | R 记录 {len(R)}")

def prep(df, acol, dcol=None):
    d = df.dropna(subset=[acol]).copy()
    d["dom"] = d["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
    d["sub"] = d["subset"].map(lambda s: "+".join(sorted(str(s).replace("+", " ").split())))
    d["pairn"] = d["pair"].astype(str).str.replace("×", "x").str.strip()
    if dcol and dcol in d.columns:
        d = d[d[dcol].fillna(0) == 0]
    return d

P4 = prep(P, "ari4", "degen4")
print(f"剔除退化后 Python 记录 {len(P4)}")

# ---------- ① 留出队列选择的噪声 ----------
g = P4.groupby(["dom", "sub", "method"])["ari4"]
sd_cohort = g.std(ddof=1).dropna()
n_pair = g.size()
print(f"\n① 留出队列选择（每组 {int(n_pair.median())} 个队列对）")
print(f"   组内 SD: 中位 {sd_cohort.median():.4f} | 均值 {sd_cohort.mean():.4f} | Q90 {sd_cohort.quantile(.9):.4f}")

# R 侧交叉验证
R4 = prep(R, "ari_4")
gR = R4.groupby(["dom", "sub", "method"])["ari_4"]
sd_cohort_R = gR.std(ddof=1).dropna()
print(f"   R 侧独立复核: 中位 {sd_cohort_R.median():.4f} | 均值 {sd_cohort_R.mean():.4f}")

# ---------- ② K 选择噪声 ----------
rec = P4.copy()
ks = rec[["ari3", "ari4", "ari5"]].values
sd_k = np.nanstd(ks, axis=1, ddof=1)
print(f"\n② K 选择（K=3/4/5）")
print(f"   记录级 SD: 中位 {np.nanmedian(sd_k):.4f} | 均值 {np.nanmean(sd_k):.4f}")

# ---------- ③ 方法配对差分布 ----------
rows = []
for (dom, sub, pr), grp in P4.groupby(["dom", "sub", "pairn"]):
    ms = grp.set_index("method")["ari4"]
    for i, a in enumerate(ms.index):
        for b in ms.index[i+1:]:
            rows.append({"dom": dom, "sub": sub, "pair": pr, "m_a": a, "m_b": b,
                         "d": ms[a] - ms[b]})
DD = pd.DataFrame(rows)
print(f"\n③ 方法配对差（{len(DD)} 对）")
print(f"   |ΔARI| 中位 {DD['d'].abs().median():.4f} | SD {DD['d'].std(ddof=1):.4f}")

# ---------- ④ 选择偏倚：基线是 max-of-k ----------
print(f"\n④ 选择偏倚（基线取 max，存在乐观偏倚）")
per_rec = P4.groupby(["dom", "sub", "pairn"])["method"].nunique()
k_methods = int(per_rec.median())
print(f"   每记录方法数 k = {k_methods}")
# 蒙特卡洛：k 个同分布方法的最大值超出真值多少
sim = []
for _ in range(20000):
    x = rng.normal(0, sd_cohort.median(), k_methods)
    sim.append(x.max() - 0)
sel_bias = float(np.mean(sim))
print(f"   max-of-{k_methods} 乐观偏倚 ≈ {sel_bias:.4f}（以组内 SD {sd_cohort.median():.4f} 为尺度）")

# ---------- 反推阈值 ----------
print("\n" + "=" * 84)
print("反推 G2 阈值")
print("=" * 84)

# 评估集规模：留出队列上一次评估覆盖的 (域,层子集) 记录数
n_eval_A = int(P4[P4["dom"] == "A"].groupby(["sub"]).size().shape[0]) if True else 0
subs_A = P4[P4["dom"] == "A"]["sub"].nunique(); subs_B = P4[P4["dom"] == "B"]["sub"].nunique()
print(f"评估集规模参考：域A {subs_A} 个层子集 | 域B {subs_B} 个 | 合计 {subs_A+subs_B}")

base = P4.groupby("method")["ari4"].mean().sort_values(ascending=False)
print(f"\n最佳基线（记录加权均值）: {base.index[0]} = {base.iloc[0]:.4f}")
print(f"次优: {base.index[1]} = {base.iloc[1]:.4f}  → 观测差值仅 {base.iloc[0]-base.iloc[1]:.4f}")
obs_spread = base.iloc[0] - base[~base.index.str.contains("SNF")].iloc[-1] if False else None
print(f"全部方法极差: {base.iloc[0]:.4f} − {base.iloc[-1]:.4f} = {base.iloc[0]-base.iloc[-1]:.4f}")

for n_eval in [subs_A, subs_A + subs_B, 2 * (subs_A + subs_B)]:
    for sd_used, tag in [(DD['d'].std(ddof=1), "配对差SD"), (sd_cohort.median(), "队列内SD")]:
        se = sd_used / np.sqrt(n_eval)
        mdd_paired = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se   # α=0.05 双侧, power=0.8
        r = mdd_paired / base.iloc[0]
        print(f"  n_eval={n_eval:3d} {tag:10s} SE={se:.4f}  MDD={mdd_paired:.4f}  相对阈值 = +{r*100:.1f}%"
              f"  → 绝对目标 {base.iloc[0]*(1+r):.4f}")

# 加上选择偏倚后的总阈值
print("\n加入选择偏倚（新算法须超过 max-of-k 的乐观部分）:")
for n_eval in [subs_A, subs_A + subs_B]:
    se = DD['d'].std(ddof=1) / np.sqrt(n_eval)
    mdd = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se
    total = mdd + sel_bias
    print(f"  n_eval={n_eval:3d}: MDD={mdd:.4f} + 偏倚={sel_bias:.4f} = {total:.4f}"
          f"  → 相对阈值 +{total/base.iloc[0]*100:.1f}%  绝对目标 {base.iloc[0]+total:.4f}")

out = {"sd_cohort_median": float(sd_cohort.median()), "sd_cohort_mean": float(sd_cohort.mean()),
       "sd_cohort_R_median": float(sd_cohort_R.median()),
       "sd_K_median": float(np.nanmedian(sd_k)),
       "sd_paired": float(DD['d'].std(ddof=1)), "n_pairs": int(len(DD)),
       "sel_bias": sel_bias, "k_methods": k_methods,
       "baseline_best": float(base.iloc[0]), "baseline_best_name": str(base.index[0]),
       "baseline_second": float(base.iloc[1]), "subs_A": int(subs_A), "subs_B": int(subs_B)}
json.dump(out, open(f"{BASE}/g2_threshold.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 g2_threshold.json")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m2：G2 阈值的决定性检验
① 胜负翻转：同一 (域,层子集) 上，冠军是否随留出队列改变？
② 达标所需样本量：要让 +10% 可检测，需要多少评估记录？
③ 备选 G2 设计（可复制性准则）的可行性
"""
import json
import numpy as np, pandas as pd
from scipy import stats

BASE = "/tmp/gyn_retyping"
G2 = json.load(open(f"{BASE}/g2_threshold.json"))
rng = np.random.RandomState(49)

P = pd.DataFrame(json.load(open(f"{BASE}/bench_matrix.json")))
for k in ["3", "4", "5"]:
    P[f"ari{k}"] = P["ari"].map(lambda d: d.get(k) if isinstance(d, dict) else None)
P["degen4"] = P["degen"].map(lambda d: d.get("4") if isinstance(d, dict) else None)
P = P.dropna(subset=["ari4"])
P = P[P["degen4"].fillna(0) == 0]
P["dom"] = P["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
P["sub"] = P["subset"].map(lambda s: "+".join(sorted(str(s).replace("+", " ").split())))
P["pairn"] = P["pair"].astype(str).str.replace("×", "x").str.strip()
print(f"分析记录 {len(P)} | 域A/域B 层子集数 {P[P.dom=='A']['sub'].nunique()}/{P[P.dom=='B']['sub'].nunique()}")

base = P.groupby("method")["ari4"].mean().sort_values(ascending=False)
BEST = base.index[0]; BEST_V = base.iloc[0]
print(f"\n最佳基线: {BEST} = {BEST_V:.4f}")

# ---------- ① 胜负翻转 ----------
print("\n" + "=" * 84)
print("① 冠军是否随留出队列翻转")
print("=" * 84)
champ = {}
for (dom, sub, pr), g in P.groupby(["dom", "sub", "pairn"]):
    champ[(dom, sub, pr)] = g.loc[g["ari4"].idxmax(), "method"]
C = pd.Series(champ)
flip = []
for (dom, sub), g in C.groupby(level=[0, 1]):
    vs = list(g.values)
    flip.append({"dom": dom, "sub": sub, "n_pairs": len(vs), "n_unique_winner": len(set(vs)),
                 "consistent": len(set(vs)) == 1, "winners": " | ".join(vs)})
FL = pd.DataFrame(flip)
print(f"共 {len(FL)} 个 (域×层子集) 单元，各含 {int(FL['n_pairs'].median())} 个留出队列对")
print(f"  冠军完全一致的单元: {int(FL['consistent'].sum())}/{len(FL)} = {FL['consistent'].mean()*100:.1f}%")
print(f"  冠军不一致（翻转）: {int((~FL['consistent']).sum())}/{len(FL)} = {(~FL['consistent']).mean()*100:.1f}%")
print(f"  平均每单元不同冠军数: {FL['n_unique_winner'].mean():.2f}")
print("\n  各方法被选为冠军的次数（共 %d 个单元×队列对）:" % int(FL['n_pairs'].sum()))
print("  " + C.value_counts().to_string().replace("\n", "\n  "))
FL.to_csv(f"{BASE}/g2_champ_flip.csv", index=False)

# ---------- ② 达标所需样本量 ----------
print("\n" + "=" * 84)
print("② 要让给定相对阈值可检测，需要多少评估记录")
print("=" * 84)
sd_paired = G2["sd_paired"]; sd_cohort = G2["sd_cohort_median"]
print(f"  噪声尺度：配对差 SD = {sd_paired:.4f} | 队列内 SD = {sd_cohort:.4f}")
print(f"  {('目标相对阈值'):>14s} {'绝对差':>9s} {'需 n(配对差)':>13s} {'需 n(队列内)':>12s}")
req = []
for r in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    delta = r * BEST_V
    n1 = (2.8 * sd_paired / delta) ** 2
    n2 = (2.8 * sd_cohort / delta) ** 2
    req.append({"rel_threshold": r, "abs_delta": round(delta, 4),
                "n_needed_paired": int(np.ceil(n1)), "n_needed_cohort": int(np.ceil(n2))})
    print(f"  {('+' + str(int(r*100)) + '%'):>14s} {delta:9.4f} {int(np.ceil(n1)):13d} {int(np.ceil(n2)):12d}")
print(f"\n  现有评估集规模: 域A 7 + 域B 15 = 22 个 (域×层子集) 单元")
print(f"  → +10% 在现有规模下**不可检测**（需 {int(np.ceil((2.8*sd_cohort/(0.10*BEST_V))**2))}–{int(np.ceil((2.8*sd_paired/(0.10*BEST_V))**2))} 条）")
pd.DataFrame(req).to_csv(f"{BASE}/g2_required_n.csv", index=False)

# ---------- ③ 可复制性准则的可行性 ----------
print("\n" + "=" * 84)
print("③ 备选 G2 设计：可复制性准则的可行性")
print("=" * 84)
print("  准则：新算法在**固定的**留出队列上，于 ≥2/3 的队列对中排名第 1（而非单队列百分比）")
print("  依据：冠军翻转率实测如下")
top2 = base.head(2).index.tolist()
print(f"  当前前 2 名: {top2[0]} ({base.iloc[0]:.4f}) / {top2[1]} ({base.iloc[1]:.4f})，差 {base.iloc[0]-base.iloc[1]:.4f}")
# 对每个方法：在多少个 (域,子集) 单元里，它在 ≥2/3 队列对中排第 1
rec = []
for m in base.index:
    sub_df = P[P["method"] == m]
    ok = 0; tot = 0
    for (dom, sub), g in sub_df.groupby(["dom", "sub"]):
        w = 0; n = 0
        for pr, gg in g.groupby("pairn"):
            allm = P[(P["dom"] == dom) & (P["sub"] == sub) & (P["pairn"] == pr)]
            if len(allm) == 0: continue
            n += 1
            if allm.loc[allm["ari4"].idxmax(), "method"] == m: w += 1
        if n >= 2:
            tot += 1
            if w / n >= 2 / 3: ok += 1
    rec.append({"method": m, "mean_ari": round(base[m], 4),
                "units_2of3_win": ok, "units_total": tot,
                "frac": round(ok / tot, 4) if tot else None})
RC = pd.DataFrame(rec).sort_values("mean_ari", ascending=False)
print("\n  " + RC.to_string(index=False).replace("\n", "\n  "))
RC.to_csv(f"{BASE}/g2_replicability.csv", index=False)
print(f"\n  → 现有基线的最高 '≥2/3 可复制夺冠率' 仅 {RC['frac'].max()*100:.1f}%"
      f"（{RC.loc[RC['frac'].idxmax(),'method']}），说明该准则**现有方法均难达成**")

out = {"champ_flip_rate": float((~FL['consistent']).mean()), "n_units": len(FL),
       "mean_unique_winners": float(FL["n_unique_winner"].mean()),
       "required_n": req, "replicability": RC.to_dict(orient="records"),
       "best_method": BEST, "best_value": float(BEST_V)}
json.dump(out, open(f"{BASE}/g2_final.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 g2_final.json")

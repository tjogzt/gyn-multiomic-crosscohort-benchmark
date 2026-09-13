#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m3：把噪声估计收窄到决策真正涉及的比较（新算法 vs 最佳基线）
并给出 G2 的两个可行冻结方案 + 功率模拟
"""
import json
import numpy as np, pandas as pd
from scipy import stats

BASE = "/tmp/gyn_retyping"
rng = np.random.RandomState(49)

P = pd.DataFrame(json.load(open(f"{BASE}/bench_matrix.json")))
for k in ["3", "4", "5"]:
    P[f"ari{k}"] = P["ari"].map(lambda d: d.get(k) if isinstance(d, dict) else None)
P["degen4"] = P["degen"].map(lambda d: d.get("4") if isinstance(d, dict) else None)
P = P.dropna(subset=["ari4"]); P = P[P["degen4"].fillna(0) == 0]
P["dom"] = P["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
P["sub"] = P["subset"].map(lambda s: "+".join(sorted(str(s).replace("+", " ").split())))
P["pairn"] = P["pair"].astype(str).str.replace("×", "x").str.strip()
P["rec"] = P["dom"] + "|" + P["sub"] + "|" + P["pairn"]

W = P.pivot_table(index="rec", columns="method", values="ari4")
print(f"宽表: {W.shape[0]} 记录 × {W.shape[1]} 方法")
base = W.mean().sort_values(ascending=False)
BEST = base.index[0]; BEST_V = base.iloc[0]
print(f"基线方法 {BEST} = {BEST_V:.4f}\n")

print("=" * 92)
print("① 逐方法对的配对差 SD（决策真正涉及的比较）")
print("=" * 92)
cand = [m for m in base.index if m != BEST][:6]
res = []
for m in cand:
    d = (m, W[BEST] - W[m]) if False else None
    dd = (W[m] - W[BEST]).dropna()
    sd = dd.std(ddof=1)
    t, p = stats.ttest_rel(W[m].dropna()[dd.index], W[BEST].reindex(dd.index))
    res.append({"vs": m, "mean_diff": round(dd.mean(), 4), "sd_diff": round(sd, 4),
                "n": len(dd), "paired_p": float(p)})
    print(f"  {m:22s} 均值差 {dd.mean():+.4f}  SD {sd:.4f}  n={len(dd)}  配对 p={p:.3g}")
SD_TOP = pd.DataFrame(res).sort_values("sd_diff")
print(f"\n  与最佳基线的配对标 SD 范围: {SD_TOP['sd_diff'].min():.4f} – {SD_TOP['sd_diff'].max():.4f}")
SD_KMCONS = float(pd.DataFrame(res).set_index("vs").loc["6_共联共识", "sd_diff"])
print(f"  最接近的挑战者（共联共识）SD = {SD_KMCONS:.4f}  ← 决策相关噪声尺度")

print("\n" + "=" * 92)
print("② 达标所需样本量（用决策相关噪声）")
print("=" * 92)
print(f"  {('相对阈值'):>10s} {'绝对差':>9s} {'需n(最接近挑战者)':>18s} {'需n(全方法对)':>14s}")
SD_ALL = json.load(open(f"{BASE}/g2_threshold.json"))["sd_paired"]
req = []
for r in [0.05, 0.10, 0.15, 0.20, 0.30]:
    delta = r * BEST_V
    n1 = (2.8 * SD_KMCONS / delta) ** 2
    n2 = (2.8 * SD_ALL / delta) ** 2
    req.append({"rel": r, "abs": round(delta, 4), "n_relevant": int(np.ceil(n1)), "n_allpairs": int(np.ceil(n2))})
    print(f"  {('+' + str(int(r*100)) + '%'):>10s} {delta:9.4f} {int(np.ceil(n1)):18d} {int(np.ceil(n2)):14d}")

print("\n" + "=" * 92)
print("③ 现有规模下的最小可检测相对变化（MDD）")
print("=" * 92)
for n in [22, 44, 66]:
    for sd, tag in [(SD_KMCONS, "最接近挑战者SD"), (SD_ALL, "全方法对SD")]:
        mdd = 2.8 * sd / np.sqrt(n)
        print(f"  n={n:3d} {tag:16s} MDD={mdd:.4f} → 相对阈值 +{mdd/BEST_V*100:.1f}%  (绝对目标 {BEST_V+mdd:.4f})")

print("\n" + "=" * 92)
print("④ 功率模拟：新算法真实提升 r_true，在 n=22 下通过 +10% 门的概率")
print("=" * 92)
print("  （假设新算法与基线的配对差 SD = 最接近挑战者的 SD）")
for r_true in [0.0, 0.10, 0.145, 0.20, 0.30]:
    true_delta = r_true * BEST_V
    hits = 0
    for _ in range(4000):
        dd = rng.normal(true_delta, SD_KMCONS, 22)
        # 单侧配对 t 检验，α=0.05
        tstat = dd.mean() / (dd.std(ddof=1) / np.sqrt(22))
        if tstat > stats.t.ppf(0.95, 21):
            hits += 1
    print(f"  真实提升 +{int(r_true*100):3d}%  →  在 n=22 上被判定为显著的概率 = {hits/4000*100:5.1f}%")

json.dump({"baseline": BEST, "baseline_value": float(BEST_V), "sd_relevant": SD_KMCONS,
           "sd_allpairs": float(SD_ALL), "candidate_pairs": res, "required_n": req},
          open(f"{BASE}/g2_refined.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 g2_refined.json")

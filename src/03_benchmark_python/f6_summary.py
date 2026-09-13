#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-6：统一口径（记录加权）的最终汇总表，供报告引用"""
import json, numpy as np
from collections import defaultdict
R = json.load(open("/tmp/gyn_retyping/bench_matrix.json"))
AV = json.load(open("/tmp/gyn_retyping/bench_availability.json"))
K = "4"
ok = [x for x in R if x.get("ari") and x["ari"].get(K) is not None]
ms = sorted({x["method"] for x in ok})
print("口径：方法/子集均值 = 跨全部 (层子集 × 队列对) 记录等权平均\n")

print("=" * 96); print("表1 方法 × 域（记录加权迁移 ARI, K=4）"); print("=" * 96)
print(f"{'方法':<20}{'域A':>10}{'域B':>10}{'合并':>10}{'最大簇占比':>12}{'退化率':>9}{'n':>5}")
S1 = {}
for m in ms:
    r = [x for x in ok if x["method"] == m]
    a = [x["ari"][K] for x in r if x["domain"] == "A_CPTAC三队列"]
    b = [x["ari"][K] for x in r if x["domain"] == "B_EC四层"]
    bal = np.mean([max(x["balance"][K]) for x in r if x.get("balance") and x["balance"].get(K)])
    dg = np.mean([x["degen"][K] for x in r if x.get("degen") and x["degen"].get(K) is not None])
    S1[m] = {"A": round(float(np.mean(a)), 4), "B": round(float(np.mean(b)), 4),
             "all": round(float(np.mean([x["ari"][K] for x in r])), 4),
             "bal": round(float(bal), 3), "deg": round(float(dg), 3), "n": len(r)}
    print(f"{m.split('_',1)[1]:<20}{S1[m]['A']:>10.4f}{S1[m]['B']:>10.4f}{S1[m]['all']:>10.4f}"
          f"{S1[m]['bal']:>12.3f}{S1[m]['deg']:>9.2f}{S1[m]['n']:>5}")
rank = sorted(ms, key=lambda m: -S1[m]["all"])
print(f"\n排名（合并 ARI）：" + " > ".join(f"{m.split('_',1)[1]}({S1[m]['all']:.3f})" for m in rank))

print("\n" + "=" * 96); print("表2 层数 → 最佳 ARI（记录最大）与可用样本"); print("=" * 96)
S2 = {}
for dom in ["A_CPTAC三队列", "B_EC四层"]:
    r = [x for x in ok if x["domain"] == dom]
    subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
    print(f"\n▍{dom}")
    print(f"{'层数':<5}{'子集数':>7}{'最佳ARI':>10}   最优子集 × 方法")
    S2[dom] = []
    for n in sorted({len(s.split("+")) for s in subs}):
        ss = [s for s in subs if len(s.split("+")) == n]
        best = None
        for s in ss:
            for m in ms:
                v = [x["ari"][K] for x in r if x["subset"] == s and x["method"] == m]
                if v and (best is None or np.mean(v) > best[2]): best = (s, m, float(np.mean(v)))
        print(f"{n:<5}{len(ss):>7}{best[2]:>10.4f}   {best[0]} × {best[1].split('_',1)[1]}")
        S2[dom].append({"nL": n, "nsub": len(ss), "best": round(best[2], 4),
                        "subset": best[0], "method": best[1].split("_", 1)[1]})

print("\n" + "=" * 96); print("表3 固定最优方法 KMeans(concat) 的层数曲线"); print("=" * 96)
BM = "3_KMeans(concat)"
S3 = {}
for dom in ["A_CPTAC三队列", "B_EC四层"]:
    r = [x for x in ok if x["domain"] == dom and x["method"] == BM]
    print(f"\n▍{dom}")
    S3[dom] = []
    for n in sorted({len(x["subset"].split("+")) for x in r}):
        v = [x["ari"][K] for x in r if len(x["subset"].split("+")) == n]
        na = [x["nA"] for x in r if len(x["subset"].split("+")) == n]
        print(f"   {n} 层 | n对 {len(v):2d} | ARI {np.mean(v):.4f} | 最大 {max(v):.4f} | 样本均值 {int(np.mean(na))}")
        S3[dom].append({"nL": n, "n": len(v), "ari": round(float(np.mean(v)), 4),
                        "max": round(float(max(v)), 4), "nA": int(np.mean(na))})

print("\n" + "=" * 96); print("表4 层数 → 可用样本（每队列，层间交集）"); print("=" * 96)
S4 = defaultdict(dict)
for x in AV: S4[x["cohort"]][(len(x["subset"].split("+")), x["subset"])] = x["n"]
for coh in ["TCGA", "ind", "dis", "OV"]:
    if coh not in S4: continue
    print(f"\n  ▍{coh}")
    for (n, s), v in sorted(S4[coh].items()):
        print(f"      {n} 层  {s:<28} n={v}")

json.dump({"tab1": S1, "tab2": S2, "tab3": S3,
           "tab4": {k: {f"{a}|{b}": c for (a, b), c in v.items()} for k, v in S4.items()}},
          open("/tmp/gyn_retyping/bench_summary.json", "w"), indent=1, ensure_ascii=False)
print("\n已落盘 bench_summary.json")

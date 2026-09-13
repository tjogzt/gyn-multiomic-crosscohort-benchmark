#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-8：PAC 聚合 + 与跨队列 ARI 的交叉分析"""
import json, numpy as np
from collections import defaultdict
P = json.load(open("/tmp/gyn_retyping/pac_matrix.json"))
A = json.load(open("/tmp/gyn_retyping/bench_matrix.json"))
ok = [x for x in P if x.get("pac") is not None]
K = "4"
AOK = [x for x in A if x.get("ari") and x["ari"].get(K) is not None]

print("=" * 104); print("【1】方法 × PAC（K=4，队列内聚类稳定性；PAC 越低越稳）"); print("=" * 104)
ms = sorted({x["method"] for x in ok})
print(f"{'方法':<20}{'域A PAC':>10}{'域B PAC':>10}{'合并 PAC':>10}{'强共识':>9}{'中位共识':>10}{'n':>5}")
T1 = {}
for m in ms:
    r = [x for x in ok if x["method"] == m]
    a = [x["pac"] for x in r if x["domain"] == "A_CPTAC三队列"]
    b = [x["pac"] for x in r if x["domain"] == "B_EC四层"]
    T1[m] = {"A": round(float(np.mean(a)), 4), "B": round(float(np.mean(b)), 4),
             "all": round(float(np.mean([x["pac"] for x in r])), 4),
             "strong": round(float(np.mean([x["strong"] for x in r])), 3),
             "med": round(float(np.mean([x["median_cons"] for x in r])), 3), "n": len(r)}
    print(f"{m.split('_',1)[1]:<20}{T1[m]['A']:>10.4f}{T1[m]['B']:>10.4f}{T1[m]['all']:>10.4f}"
          f"{T1[m]['strong']:>9.3f}{T1[m]['med']:>10.3f}{T1[m]['n']:>5}")
rk = sorted(ms, key=lambda m: T1[m]["all"])
print(f"\n  最稳 → 最不稳: " + " < ".join(f"{m.split('_',1)[1]}({T1[m]['all']:.3f})" for m in rk))

print("\n" + "=" * 104); print("【2】层子集 × PAC（跨队列平均；按层数归类）"); print("=" * 104)
for dom in ["A_CPTAC三队列", "B_EC四层"]:
    r = [x for x in ok if x["domain"] == dom]
    subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
    print(f"\n▍{dom}")
    for n in sorted({len(s.split("+")) for s in subs}):
        ss = [s for s in subs if len(s.split("+")) == n]
        v = [x["pac"] for x in r if x["subset"] in ss]
        print(f"   {n} 层 | 子集 {len(ss):2d} | PAC 均值 {np.mean(v):.4f} | 最稳 {min(v):.4f} | 最不稳 {max(v):.4f}")

print("\n" + "=" * 104); print("【3】PAC × 跨队列 ARI 的交叉（核心问题：更稳是否更可迁移？）"); print("=" * 104)
# PAC 按 (域, 队列, 子集, 方法)；ARI 按 (域, 子集, 队列对, 方法)
pacd = {}
for x in ok:
    pacd[(x["domain"], x["cohort"], x["subset"], x["method"])] = x["pac"]
pairs = []
for x in AOK:
    d, sub, pr, m = x["domain"], x["subset"], x["pair"], x["method"]
    a, b = pr.split("×")
    pa = pacd.get((d, a, sub, m)); pb = pacd.get((d, b, sub, m))
    if pa is not None and pb is not None:
        pairs.append({"domain": d, "subset": sub, "pair": pr, "method": m,
                      "ari": x["ari"][K], "pac": (pa + pb) / 2,
                      "pacA": pa, "pacB": pb, "nA": x["nA"], "nB": x["nB"]})
print(f"  可交叉记录: {len(pairs)}")
if pairs:
    ari = np.array([p["ari"] for p in pairs]); pac = np.array([p["pac"] for p in pairs])
    from scipy.stats import spearmanr, pearsonr
    print(f"  ★ Spearman(ARI, PAC)  = {spearmanr(ari, pac)[0]:+.4f}  (p={spearmanr(ari,pac)[1]:.2e})")
    print(f"    Pearson (ARI, PAC)  = {pearsonr(ari, pac)[0]:+.4f}  (p={pearsonr(ari,pac)[1]:.2e})")
    print(f"    → 负相关 = 越稳越可迁移；正相关 = 稳与可迁移无关或反向")
    # 分域
    for dom in ["A_CPTAC三队列", "B_EC四层"]:
        s = [p for p in pairs if p["domain"] == dom]
        if len(s) > 10:
            a2 = np.array([p["ari"] for p in s]); p2 = np.array([p["pac"] for p in s])
            print(f"      {dom}: n={len(s)}  Spearman {spearmanr(a2,p2)[0]:+.4f} (p={spearmanr(a2,p2)[1]:.2e})")
    # PAC 分箱
    print("\n  ▍按 PAC 分箱看 ARI")
    for lo, hi in [(0, .3), (.3, .45), (.45, .6), (.6, 1.01)]:
        s = [p for p in pairs if lo <= p["pac"] < hi]
        if s: print(f"      PAC [{lo:.2f},{hi:.2f}) n={len(s):3d}  ARI 均值 {np.mean([p['ari'] for p in s]):.4f}")
    # 最稳 / 最不稳 子集
    print("\n  ▍最稳 vs 最不稳 层子集（按 PAC）")
    for dom in ["A_CPTAC三队列", "B_EC四层"]:
        s = [p for p in pairs if p["domain"] == dom]
        if not s: continue
        ag = defaultdict(lambda: [0, 0])
        for p in s: ag[p["subset"]][0] += p["pac"]; ag[p["subset"]][1] += 1
        ss = sorted(ag, key=lambda k: ag[k][0] / ag[k][1])
        print(f"      {dom}: 最稳 {ss[0]} (PAC {ag[ss[0]][0]/ag[ss[0]][1]:.3f}) | 最不稳 {ss[-1]} (PAC {ag[ss[-1]][0]/ag[ss[-1]][1]:.3f})")

json.dump({"tab1": T1, "cross": pairs}, open("/tmp/gyn_retyping/pac_agg.json", "w"),
          indent=1, ensure_ascii=False)
print("\n已落盘 pac_agg.json")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-4：最终聚合 —— 方法×子集 迁移 ARI + 边际价值 + 退化/恒等诊断"""
import json, numpy as np
from collections import defaultdict
R = json.load(open("/tmp/gyn_retyping/bench_matrix.json"))
A = json.load(open("/tmp/gyn_retyping/bench_availability.json"))
K = 4
ok = [x for x in R if x.get("ari") and x["ari"].get(str(K)) is not None]

print("=" * 104); print("【1】方法总体表现（K=4，域A+域B 合并）"); print("=" * 104)
for m in sorted({x["method"] for x in ok}):
    r = [x for x in ok if x["method"] == m]
    ari = np.mean([x["ari"][str(K)] for x in r])
    bal = np.mean([max(x["balance"][str(K)]) for x in r if x.get("balance") and x["balance"].get(str(K))])
    dg = np.mean([x["degen"][str(K)] for x in r if x.get("degen") and x["degen"].get(str(K)) is not None])
    print(f"  {m:<20} ARI均值 {ari:.4f} | 最大簇占比 {bal:.3f} | 退化率 {dg:.2f} | n={len(r)}")

print("\n" + "=" * 104); print("【2】方法间的恒等/近恒等对（逐记录比对）"); print("=" * 104)
ms = sorted({x["method"] for x in ok})
key = lambda x: (x["domain"], x["subset"], x["pair"])
by = defaultdict(dict)
for x in ok: by[key(x)][x["method"]] = x["ari"][str(K)]
for i in range(len(ms)):
    for j in range(i+1, len(ms)):
        d = [abs(v[ms[i]]-v[ms[j]]) for v in by.values() if ms[i] in v and ms[j] in v]
        if len(d) >= 5:
            same = sum(1 for t in d if t < 1e-9)
            print(f"  {ms[i]:<20} vs {ms[j]:<20} 可比 {len(d):3d} | 最大差 {max(d):.4f} | 完全相同 {same}/{len(d)}")

for dom in ["A_CPTAC三队列", "B_EC四层"]:
    r = [x for x in ok if x["domain"] == dom]
    if not r: continue
    subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
    meths = sorted({x["method"] for x in r})
    print(f"\n{'='*104}\n【3】{dom} · 层子集 × 方法（K=4，跨队列对均值）\n{'='*104}")
    print(f"{'层子集':<26}{'nL':>3}{'n对':>4}  " + "".join(f"{m.split('_')[1][:8]:>10}" for m in meths))
    agg = defaultdict(list); keep = {}
    for s in subs:
        line = f"{s:<26}{len(s.split('+')):>3}"
        npair = len({x["pair"] for x in r if x["subset"] == s})
        line += f"{npair:>4}  "
        row = {}
        for m in meths:
            v = [x["ari"][str(K)] for x in r if x["subset"] == s and x["method"] == m]
            row[m] = round(float(np.mean(v)), 4) if v else None
            agg[m].append(row[m])
            line += f"{(f'{row[m]:.3f}' if row[m] is not None else '—'):>10}"
        keep[s] = row
        print(line)
    print(f"\n  ▸ 各方法平均（跨子集）")
    for m in meths:
        vs = [v for v in agg[m] if v is not None]
        if vs: print(f"      {m:<20} {np.mean(vs):.4f}")
    print(f"\n  ▸ 边际价值：层数 → 最佳 ARI / 最优子集")
    for nl in sorted({len(s.split('+')) for s in subs}):
        ss = [s for s in subs if len(s.split('+')) == nl]
        best = None
        for s in ss:
            for m, v in keep[s].items():
                if v is not None and (best is None or v > best[2]): best = (s, m, v)
        if best: print(f"      {nl} 层 | 子集 {len(ss):2d} 个 | 最佳 {best[2]:.4f}  ← {best[0]} × {best[1].split('_')[1]}")
    json.dump(keep, open(f"/tmp/gyn_retyping/bench_table_{dom[:1]}.json", "w"), indent=1, ensure_ascii=False)

print("\n" + "=" * 104); print("【4】逐层边际贡献（在最优基线之上加层的净增益，K=4）"); print("=" * 104)
for dom in ["A_CPTAC三队列", "B_EC四层"]:
    r = [x for x in ok if x["domain"] == dom]
    if not r: continue
    meths = sorted({x["method"] for x in r})
    best_m = max(meths, key=lambda m: np.mean([x["ari"][str(K)] for x in r if x["method"] == m]))
    print(f"\n  ▍{dom}  最优方法 = {best_m}")
    subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
    for s in subs:
        v = [x["ari"][str(K)] for x in r if x["subset"] == s and x["method"] == best_m]
        n = [x["nA"] for x in r if x["subset"] == s and x["method"] == best_m]
        if v: print(f"      {s:<26} ARI {np.mean(v):.4f} | nA均值 {int(np.mean(n))}")

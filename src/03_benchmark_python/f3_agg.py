#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-3：基准聚合 —— 层子集 × 方法 迁移 ARI、边际价值曲线、覆盖率"""
import json, numpy as np
from collections import defaultdict
R = json.load(open("/tmp/gyn_retyping/bench_matrix.json"))
A = json.load(open("/tmp/gyn_retyping/bench_availability.json"))
K = "4"
print("=" * 100); print("【覆盖率】"); print("=" * 100)
for dom in sorted({x["domain"] for x in R}):
    r = [x for x in R if x["domain"] == dom]
    ok = [x for x in r if x.get("ari") and x["ari"].get(K) is not None]
    err = [x for x in r if x.get("err")]
    from collections import Counter
    ec = Counter(x["err"][:34] for x in err)
    print(f"  {dom}: 记录 {len(r)} | 有效 {len(ok)} | 错误 {len(err)}")
    for k, v in ec.most_common(4): print(f"      {v:3d}× {k}")

# 方法 × 层子集（均值）
print("\n" + "=" * 100); print(f"【域A · 层子集 × 方法 迁移 ARI (K={K})】"); print("=" * 100)
for dom in ["A_CPTAC三队列", "B_EC四层"]:
    r = [x for x in R if x["domain"] == dom and x.get("ari") and x["ari"].get(K) is not None]
    if not r: continue
    subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
    meths = sorted({x["method"] for x in r})
    agg = defaultdict(list)
    for x in r: agg[(x["subset"], x["method"])].append(x["ari"][K])
    print(f"\n▍{dom}")
    print(f"{'层子集':<26}{'层数':>4}{'n对':>5}  " + "".join(f"{m.split('_')[1][:9]:>11}" for m in meths) + f"{'最佳':>14}")
    BEST = {}
    for s in subs:
        vals = {m: (np.mean(agg[(s, m)]) if agg.get((s, m)) else None) for m in meths}
        npair = max(len(agg[(s, m)]) for m in meths)
        line = f"{s:<26}{len(s.split('+')):>4}{npair:>5}  "
        for m in meths:
            v = vals[m]; line += f"{(f'{v:.3f}' if v is not None else '—'):>11}"
        ok = {m: v for m, v in vals.items() if v is not None}
        bm = max(ok, key=ok.get) if ok else None
        line += f"{(bm.split('_')[1][:10] + ' ' + f'{ok[bm]:.3f}') if bm else '—':>14}"
        BEST[s] = (bm, ok.get(bm) if bm else None)
        print(line)
    # 每方法的平均表现
    print(f"\n  ▸ 方法平均（跨全部子集）")
    for m in meths:
        vs = [agg[(s, m)] for s in subs if agg.get((s, m))]
        if vs: print(f"      {m:<20} {np.mean([np.mean(v) for v in vs]):.4f}  (子集数 {len(vs)})")
    # 边际价值：层数 vs 最佳 ARI
    print(f"\n  ▸ 边际价值：层数 → 最佳 ARI")
    for nl in sorted({len(s.split('+')) for s in subs}):
        ss = [s for s in subs if len(s.split('+')) == nl]
        vs = [BEST[s][1] for s in ss if BEST[s][1] is not None]
        if vs:
            bs = max(ss, key=lambda s: BEST[s][1] if BEST[s][1] is not None else -1)
            print(f"      {nl} 层 | 子集数 {len(ss):2d} | 最佳 ARI {max(vs):.4f} | 最优子集 {bs}")
    json.dump({s: {"best_method": BEST[s][0], "best_ari": BEST[s][1]} for s in subs},
              open(f"/tmp/gyn_retyping/bench_best_{dom[:1]}.json", "w"), indent=1, ensure_ascii=False)

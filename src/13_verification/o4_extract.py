#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""o4：Methods/Results 所需的全部数字——从产物逐项提取"""
import json, os
import numpy as np, pandas as pd
T = "/tmp/gyn_retyping"
def j(p, d=None):
    p = os.path.join(T, p)
    return json.load(open(p)) if os.path.exists(p) else d

print("=" * 94); print("【1】队列与层可用性"); print("=" * 94)
inv = j("harmonized/inventory.json")
if inv:
    if isinstance(inv, dict):
        for k, v in list(inv.items())[:3]: print(f"  {k}: {str(v)[:160]}")
    else: print(pd.DataFrame(inv).head(12).to_string())
core5 = j("ucec_core5.json")
if core5: print(f"\n  ucec_core5: {type(core5).__name__} " + (str({k: (len(v) if hasattr(v,'__len__') else v) for k,v in core5.items()}) if isinstance(core5, dict) else str(len(core5))))
ns = j("ucec_core_nsmp.json")
if ns: print(f"  ucec_core_nsmp: " + str({k: len(v) for k, v in ns.items()}))

print("\n" + "=" * 94); print("【2】跨队列对齐（F-13~F-23）"); print("=" * 94)
ac = j("align_concordance.json")
if ac:
    print("  align_concordance:", json.dumps(ac, ensure_ascii=False)[:1400])
ad = j("align_domains.json")
if ad: print("\n  align_domains:", json.dumps(ad, ensure_ascii=False)[:900])
ai = j("align_ids.json")
if ai: print("\n  align_ids:", json.dumps(ai, ensure_ascii=False)[:700])
b3 = j("b3_anchors.json")
if b3: print("\n  b3_anchors:", json.dumps(b3, ensure_ascii=False)[:700])
b3c = j("bridge_3cohort.json")
if b3c: print("\n  bridge_3cohort:", json.dumps(b3c, ensure_ascii=False)[:900])
bd = j("bridge_diag.json")
if bd: print("\n  bridge_diag:", json.dumps(bd, ensure_ascii=False)[:700])

print("\n" + "=" * 94); print("【3】AR 对齐策略矩阵（G4）"); print("=" * 94)
ar = j("ari_agg.json")
if ar: print(" ", json.dumps(ar, ensure_ascii=False)[:1200])

print("\n" + "=" * 94); print("【4】基准（表 1/2 与层数曲线）"); print("=" * 94)
bs = j("bench_summary.json")
if bs: print(" ", json.dumps(bs, ensure_ascii=False)[:1600])
for f in ["bench_table_A.json", "bench_table_B.json"]:
    v = j(f)
    if v: print(f"\n  {f}:", json.dumps(v, ensure_ascii=False)[:800])

print("\n" + "=" * 94); print("【5】PAC（表 5-8）"); print("=" * 94)
pa = j("pac_agg.json")
if pa:
    print("  tab1:", json.dumps(pa["tab1"], ensure_ascii=False))
    cr = pa.get("cross")
    if isinstance(cr, list):
        C = pd.DataFrame(cr)
        p = pd.to_numeric(C["pac"], errors="coerce"); a = pd.to_numeric(C["ari"], errors="coerce")
        bins = pd.cut(p, [0, .30, .45, .60, 1.01], right=False)
        print("\n  PAC 分箱 → ARI:")
        print(C.assign(a=a, b=bins).dropna(subset=["a"]).groupby("b", observed=True)["a"].agg(["mean","count"]).round(4).to_string())

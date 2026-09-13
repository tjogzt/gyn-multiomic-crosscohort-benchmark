#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C-3b：ARI 聚合（去重）+ 恒等验证 + 方向统计"""
import json, numpy as np
from collections import defaultdict
res = json.load(open("/tmp/gyn_retyping/ari_matrix.json"))
STR = ["S0_基线", "S1_分位数标准化", "S2_Δ参考校正", "S3_ComBat(池化)"]
LAYS = ["mRNA", "miRNA", "CNA", "protein", "meth"]

# 唯一化：(层, 队列对, 策略)
U = {}
for r in res:
    if not r.get("strategy"): continue
    U[(r["layer"], r["pair"], r["strategy"])] = r
print(f"唯一记录: {len(U)}（原始 {len(res)}，去重 {len(res)-len(U)}）")

KEYS = sorted({(k[0], k[1]) for k in U})
print(f"唯一 (层,队列对) 组合: {len(KEYS)}")

print("\n" + "=" * 90); print("【A】恒等性验证：各策略与 S0 基线的逐值最大差（K=3/4/5）"); print("=" * 90)
for s in ["S1_分位数标准化", "S2_Δ参考校正", "S3_ComBat(池化)"]:
    ds, n = [], 0
    for L, p in KEYS:
        a = U.get((L, p, "S0_基线")); b = U.get((L, p, s))
        if a and b and a.get("ari", {}).get("3") is not None and b.get("ari", {}).get("3") is not None:
            ds.append(max(abs(a["ari"][str(k)] - b["ari"][str(k)]) for k in (3, 4, 5))); n += 1
    if n:
        print(f"  {s:<18} 可比 {n:2d} 对 | 最大逐值差 {max(ds):.6f} | 完全相同 {sum(1 for x in ds if x < 1e-9)}/{n}")

print("\n" + "=" * 90); print("【B】层 × 策略 · 迁移 ARI（K=4，跨队列对均值）"); print("=" * 90)
agg = defaultdict(lambda: defaultdict(list))
for L, p in KEYS:
    for s in STR:
        r = U.get((L, p, s))
        if r and r.get("ari") and r["ari"].get("4") is not None:
            agg[L][s].append(r["ari"]["4"])
print(f"{'层':<9}{'对数':>5}  " + "".join(f"{s:>18}" for s in STR))
TAB = {}
for L in LAYS:
    npc = len({p for (ll, p) in KEYS if ll == L})
    row = {}; line = f"{L:<9}{npc:>5}  "
    for s in STR:
        v = agg[L].get(s)
        if v: row[s] = round(float(np.mean(v)), 4); line += f"{np.mean(v):>18.4f}"
        else: row[s] = None; line += f"{'—':>18}"
    TAB[L] = row; print(line)

print("\n" + "=" * 90); print("【C】逐队列对明细（K=4）"); print("=" * 90)
print(f"{'层':<9}{'队列对':<14}{'nA/nB':>11}{'基因':>7}  " + "".join(f"{s[:12]:>14}" for s in STR))
DET = []
for L, p in KEYS:
    base = U.get((L, p, "S0_基线"))
    row = {"layer": L, "pair": p,
           "nA": base["nA"] if base else None, "nB": base["nB"] if base else None,
           "ngene": base["ngene"] if base else None}
    line = f"{L:<9}{p:<14}{(str(row['nA'])+'/'+str(row['nB'])) if base else '—':>11}{row['ngene'] or 0:>7}  "
    for s in STR:
        r = U.get((L, p, s))
        v = r["ari"]["4"] if (r and r.get("ari") and r["ari"].get("4") is not None) else None
        row[s] = v; line += f"{(f'{v:.4f}' if v is not None else '—'):>14}"
    DET.append(row); print(line)

print("\n" + "=" * 90); print("【D】分位数标准化 S1 相对 S0 的方向一致性（K=4）"); print("=" * 90)
w = l = e = 0
for r in DET:
    a, b = r["S0_基线"], r["S1_分位数标准化"]
    if a is None or b is None: continue
    tag = "↑改善" if b > a + 0.01 else ("↓变差" if b < a - 0.01 else "≈持平")
    if tag == "↑改善": w += 1
    elif tag == "↓变差": l += 1
    else: e += 1
    print(f"  {r['layer']:<9}{r['pair']:<14}{a:.4f} → {b:.4f}   {tag}")
print(f"  → 合计：改善 {w} / 变差 {l} / 持平 {e}  （共 {w+l+e} 对）")

print("\n" + "=" * 90); print("【E】跨队列可重复性天花板（S0 基线，K=4）"); print("=" * 90)
for L in LAYS:
    vals = sorted([r["S0_基线"] for r in DET if r["layer"] == L and r["S0_基线"] is not None], reverse=True)
    if vals:
        print(f"  {L:<9} 最高 {vals[0]:.4f} | 最低 {vals[-1]:.4f} | 均值 {np.mean(vals):.4f} | 极差 {vals[0]-vals[-1]:.4f}")

json.dump({"agg_K4": TAB, "detail": DET, "direction": {"better": w, "worse": l, "same": e}},
          open("/tmp/gyn_retyping/ari_agg.json", "w"), indent=1, ensure_ascii=False)
print("\n已落盘 ari_agg.json")

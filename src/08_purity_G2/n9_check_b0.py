#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n9：核对 PAI 的 B0 是否与基准的 KMeans(concat) 在同一记录上一致"""
import json
import numpy as np, pandas as pd

BEN = pd.DataFrame(json.load(open("/tmp/gyn_retyping/bench_matrix.json")))
BEN["ari4"] = BEN["ari"].map(lambda d: d.get("4") if isinstance(d, dict) else None)
BEN["deg4"] = BEN["degen"].map(lambda d: d.get("4") if isinstance(d, dict) else None)
BEN["dom"] = BEN["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
BEN["sub"] = BEN["subset"].map(lambda s: "+".join(sorted(str(s).replace("+", " ").split())))
BEN["pairn"] = BEN["pair"].astype(str).str.replace("×", "x").str.strip()
BEN["rec"] = BEN["dom"] + "|" + BEN["sub"] + "|" + BEN["pairn"]
K = BEN[BEN["method"].astype(str).str.startswith("3_KMeans")].set_index("rec")["ari4"]

P = pd.DataFrame(json.load(open("/tmp/gyn_retyping/pai/pai_matrix.json")))
P = P[P["ari"].notna()].copy()
P["ari4"] = P["ari"].map(lambda d: d.get("4") if isinstance(d, dict) else None)
P["deg4"] = P["degen"].map(lambda d: d.get("4") if isinstance(d, dict) else None)
P["dom"] = P["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
P["sub"] = P["subset"].map(lambda s: "+".join(sorted(str(s).replace("+", " ").split())))
P["pairn"] = P["pair"].astype(str).str.replace("×", "x").str.strip()
P["rec"] = P["dom"] + "|" + P["sub"] + "|" + P["pairn"]
B0 = P[P["arm"] == "B0"].set_index("rec")[["ari4", "deg4"]]

J = pd.concat([K.rename("bench_KMeans"), B0["ari4"].rename("pai_B0")], axis=1)
print(f"基准 KMeans 记录 {len(K)} | PAI B0 记录 {len(B0)} | 共同 {J.dropna().shape[0]}")
Jc = J.dropna()
if len(Jc):
    d = Jc["pai_B0"] - Jc["bench_KMeans"]
    print(f"共同记录上的差异: 均值 {d.mean():+.6f} | 最大绝对差 {d.abs().max():.6f} | 完全一致 {int((d.abs()<1e-9).sum())}/{len(Jc)}")
    print(f"  基准均值 {Jc['bench_KMeans'].mean():.4f} | PAI B0 均值 {Jc['pai_B0'].mean():.4f}")
    if d.abs().max() > 1e-6:
        print("\n  不一致的记录（前 8）:")
        print(Jc[d.abs() > 1e-6].assign(diff=d[d.abs() > 1e-6]).head(8).round(4).to_string())
print(f"\n仅基准有的记录 {len(set(K.index) - set(B0.index))} | 仅 PAI 有的记录 {len(set(B0.index) - set(K.index))}")
print("仅基准: ", sorted(set(K.index) - set(B0.index))[:8])
print("仅 PAI: ", sorted(set(B0.index) - set(K.index))[:8])

# 退化记录情况
print(f"\n退化记录（deg4!=0）: 基准 {int((BEN['deg4'].fillna(0)!=0).sum())} | PAI B0 {int((B0['deg4'].fillna(0)!=0).sum())}")
# 剔除退化后的对照
Kf = BEN[(BEN["method"].astype(str).str.startswith("3_KMeans")) & (BEN["deg4"].fillna(0) == 0)]
print(f"基准 KMeans 剔退化后均值 = {Kf['ari4'].mean():.4f} (n={len(Kf)})")
B0f = B0[B0["deg4"].fillna(0) == 0]
print(f"PAI B0 剔退化后均值     = {B0f['ari4'].mean():.4f} (n={len(B0f)})")
Jf = pd.concat([Kf.set_index("rec")["ari4"].rename("bench"), B0f["ari4"].rename("pai")], axis=1).dropna()
print(f"同在剔退化交集上: bench {Jf['bench'].mean():.4f} vs pai {Jf['pai'].mean():.4f} (n={len(Jf)})")

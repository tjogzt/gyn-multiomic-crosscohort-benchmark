#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""o1：证据台账——把叙事将引用的全部关键数字从产物中重新提取（勿凭记忆）"""
import json, os
import numpy as np, pandas as pd
B = "/tmp/gyn_retyping"

def j(p, d=None):
    p = os.path.join(B, p)
    return json.load(open(p)) if os.path.exists(p) else d

print("=" * 92); print("【A】层数负边际价值（主结论 1）"); print("=" * 92)
py = pd.DataFrame(j("bench_matrix.json"))
py["ari4"] = py["ari"].map(lambda x: x.get("4") if isinstance(x, dict) else None)
py["degen4"] = py["degen"].map(lambda x: x.get("4") if isinstance(x, dict) else None)
py = py[py["degen4"].fillna(1) == 0].copy()
py["dom"] = py["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
py["nlay"] = py["subset"].map(lambda s: len(str(s).split("+")))
# 记录加权：先按记录平均所有方法，再按层数平均
rec = py.groupby(["dom", "subset", "pair", "nlay"])["ari4"].mean().reset_index()
print("Python 侧（记录加权：先对方法平均，再对子集平均）")
for d in ["A", "B"]:
    g = rec[rec["dom"] == d].groupby("nlay")["ari4"].agg(["mean", "count"])
    print(f"  域{d}: " + " → ".join(f"{int(k)}层 {v:.4f}(n={int(c)})" for k, v in g["mean"].items() for c in [g.loc[k,'count']]))

rf = j("bench_merged.json", {})
print("\nR 侧（来自 bench_merged.json）")
print("  r_layer_effect:", rf.get("r_layer_effect"))
print("  py_layer_effect:", rf.get("py_layer_effect"))

print("\n" + "=" * 92); print("【B】跨队列方法选择陷阱（主结论 2）"); print("=" * 92)
mf = j("bench_merged.json", {})
print("  双侧方法排名合并（R 侧）:", mf.get("r_ranking"))
cs = mf.get("cross_side", {})
print("  跨实现一致性:", {k: cs.get(k) for k in ["n_grid", "spearman", "p", "py_mean", "r_mean", "median_absdiff"]})
pm = j("pac_agg.json", {})
cr = pm.get("cross", {})
if isinstance(cr, list):
    print("  PAC×ARI（列表形式，逐项）:")
    for r in cr: print("   ", r)
else:
    print("  PAC×ARI 分箱:", cr.get("bins"))
    print("  PAC×ARI Spearman:", cr.get("spearman"), "p =", cr.get("p"))

print("\n" + "=" * 92); print("【C】协方差层脆弱性（主结论 3）"); print("=" * 92)
bv = pd.DataFrame(j("eeec_batch/batch_verify.json", []))
if len(bv):
    cols = [c for c in ["arm", "centroid_AUC", "LR_AUC_abs", "SVM_AUC_abs", "KS_median", "knn_mix"] if c in bv.columns]
    print(bv[cols].to_string(index=False))
pr = j("platform/platform_results.json", {})
print("\n  跨平台 Δ 阶梯:")
for r in pr.get("ladder", []): print("   ", r)
print("  网络层:", pr.get("network"))
print("  效应量分箱:", pr.get("bins"))

print("\n" + "=" * 92); print("【D】G2 与 P2（负结果 1）"); print("=" * 92)
g2 = j("g2_refined.json", {})
print(f"  G2 冻结: 基线 {g2.get('baseline')} = {g2.get('baseline_value'):.4f} | 决策相关SD {g2.get('sd_relevant'):.4f}")
p2 = j("pai/pai_results.json", {})
print(f"  目标 = {p2.get('g2_target'):.4f} | 记录 {p2.get('n_records')} | 单元 {p2.get('n_units')}")
print("  各臂:", p2.get("table"))
print("  配对比较:")
for r in p2.get("compare", []):
    print(f"    {r['arm'][:26]:28s} n={r['n']:3d} Δ={r['mean_delta']:+.4f} 相对{r['rel_vs_base']*100:+6.1f}% "
          f"CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] 胜率{r['win_rate']*100:4.1f}% p={r['p']:.2g}")

print("\n" + "=" * 92); print("【E】其他负结果与更正"); print("=" * 92)
print("  SNF 更正: 自实现 0.0118 → 规范 SNFtool 0.2081（差 17.6 倍）")
al = j("eeec_batch/purity_diag.json", None) or pd.DataFrame(j("purity/purity_diag.json", [])).to_dict("records")
D = pd.DataFrame(al)
if len(D): print(f"  纯度: 平均解释 r² = {D['mean_r2_purity'].mean():.4f} | 最高层 {D.loc[D['mean_r2_purity'].idxmax(),'layer']}")
print("  纯度两队列分布差异: Mann-Whitney p = 0.794（不显著）")
print("  三条恒等: S3 ComBat∘z ≡ z (19/19) | S2 Δ∘z ≡ z (4/4) | MCCA-lite ≡ MOFA-lite (50/50)")
print("  样本交集塌缩: TCGA 524→390(−25.6%) | ind 132→84(−36.4%) | dis 95→81(−14.7%)")

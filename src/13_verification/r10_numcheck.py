#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r10：逐图数字核对——图中显示的每个数值须与产物一致"""
import fitz, json, re, os
import pandas as pd, numpy as np
R = "/Users/taozhu/my researches/retyping"; T = "/tmp/gyn_retyping"
def J(p): return json.load(open(f"{T}/{p}"))
bs, bm = J("bench_summary.json"), J("bench_merged.json")
av = pd.DataFrame(J("bench_availability.json"))
b3c = J("bridge_3cohort.json"); b3 = J("b3_anchors.json")

def txt(f):
    d = fitz.open(f"{R}/{f}.pdf")
    t = " ".join(pg.get_text() for pg in d)
    return re.sub(r"\s+", " ", t).replace("−", "-")

def expect_avail():
    v = av.copy(); v["nL"] = v["subset"].map(lambda s: len(str(s).split("+")))
    g = v.groupby(["domain", "cohort", "nL"])["n"].max()
    return [str(int(x)) for x in g.values]

EXP = {}
# Fig1: 可得性 + 蛋白 raw/Δ + 锚
EXP["Fig1_availability_comparability"] = (
    expect_avail() +
    [f"{b3c['raw'][k]['rho']:.3f}" for k in ["dis × ov", "dis × ind", "ov × ind"]] +
    [f"{b3c['delta'][k]['rho']:.3f}" for k in ["Δdis × Δov", "Δdis × Δind", "Δov × Δind"]] +
    [f"{b3['dis_raw_Prot_RNA']['rho']:.3f}", f"{b3['dis_dProt_dRNA']['rho']:.3f}",
     f"{b3['ind_raw_Prot_RNA']['rho']:.3f}", f"{b3['ind_dProt_dRNA']['rho']:.3f}"])
# Fig2: 层数曲线（双侧）+ 交集
EXP["Fig2_layer_negative_marginal"] = (
    [f"{bs['tab3'][k][i]['ari']:.3f}" for k in ["A_CPTAC三队列", "B_EC四层"] for i in range(len(bs["tab3"][k]))] +
    [f"{bm['r_layer_effect'][k]:.3f}" for k in ["A|1", "A|2", "A|3", "B|1", "B|2", "B|3", "B|4"]] +
    expect_avail())
# Fig3: 方法排名（双侧）+ 跨实现 + PAC
EXP["Fig3_method_not_resolvable"] = (
    [f"{bs['tab1'][m]['all']:.3f}" for m in bs["tab1"]] +
    [f"{bm['r_ranking'][m]:.3f}" for m in bm["r_ranking"]] +
    ["0.418", "50"])

for f, vals in EXP.items():
    t = txt(f)
    miss = [v for v in dict.fromkeys(vals) if v not in t]
    print(f"── {f}: 期望 {len(set(vals))} 个数值，缺失 {len(miss)}")
    if miss: print(f"     缺: {miss}")

# 数值总量统计
for f in EXP:
    t = txt(f)
    nums = set(re.findall(r"-?\d+\.\d{3}", t))
    print(f"   {f}: 图内 3 位小数数值 {len(nums)} 个")

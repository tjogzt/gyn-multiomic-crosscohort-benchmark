#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""o5：Methods/Results 关键数字 → 产物交叉核验（自动抽取，非手工清单）"""
import json, re, os
import numpy as np, pandas as pd
T = "/tmp/gyn_retyping"
R = "/Users/taozhu/my researches/retyping"
def j(p):
    return json.load(open(os.path.join(T, p)))

md = {}
for f in ["F1_Methods.md", "F1_Results.md"]:
    md[f] = open(os.path.join(R, f)).read()
ALL = "\n".join(md.values())

bs, pa, ac, b3, b3c, av = (j("bench_summary.json"), j("pac_agg.json"), j("align_concordance.json"),
                           j("b3_anchors.json"), j("bridge_3cohort.json"), pd.DataFrame(j("bench_availability.json")))

checks = []
def norm(s):
    """归一化：统一减号、去尾零"""
    s = s.replace("\u2212", "-").replace("–", "-").replace("—", "-")
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s

def ck(label, val, src, fmt="{:.4f}"):
    s = fmt.format(val) if isinstance(val, (int, float, np.floating)) else str(val)
    checks.append((label, s, src, norm(s) in norm(ALL)))

# ---- 层数曲线（tab3）----
for dom, key in [("A_CPTAC三队列", "A"), ("B_EC四层", "B")]:
    for row in bs["tab3"][dom]: ck(f"层数曲线 {key} nL={row['nL']}", row["ari"], "bench_summary.tab3")
# ---- 方法表（tab1）----
for m, v in bs["tab1"].items():
    ck(f"表1 {m} all", v["all"], "bench_summary.tab1")
    ck(f"表1 {m} bal", v["bal"], "bench_summary.tab1")
    ck(f"表1 {m} deg", v["deg"], "bench_summary.tab1")
# ---- PAC ----
for m, v in pa["tab1"].items():
    ck(f"PAC {m} all", v["all"], "pac_agg.tab1")
    ck(f"PAC {m} strong", v["strong"], "pac_agg.tab1")
# ---- 对齐 ----
for lay, d in ac.items():
    for pair, v in d.items():
        ck(f"对齐 {lay} {pair}", v["rho"], "align_concordance", "{:.4f}")
for k, v in b3c["raw"].items(): ck(f"蛋白原始 {k}", v["rho"], "bridge_3cohort.raw")
for k, v in b3c["delta"].items(): ck(f"蛋白Δ {k}", v["rho"], "bridge_3cohort.delta")
for k in ["dis_dProt_dRNA", "ind_dProt_dRNA", "dis_raw_Prot_RNA", "ind_raw_Prot_RNA"]:
    ck(f"锚 {k}", b3[k]["rho"], "b3_anchors")
# ---- 样本交集 ----
d = av.copy(); d["nL"] = d["subset"].map(lambda s: len(str(s).split("+")))
g = d.groupby(["domain", "cohort", "nL"])["n"].max()
for (dom, coh, nl), n in g.items():
    checks.append((f"样本 {dom}/{coh}/{nl}层", str(int(n)), "bench_availability", str(int(n)) in ALL))

bad = [(l, s, s2) for l, s, s2, ok in checks if not ok]
print(f"核验项总数: {len(checks)}  通过: {len(checks)-len(bad)}  未命中: {len(bad)}\n")
if bad:
    print("未命中（需人工判断是排版差异还是数字错误）:")
    for l, s, s2 in bad: print(f"  ✗ {l:44s} 产物值={s:>10s}  ({s2})")
else:
    print("✅ 全部命中")

# ---- 派生值抽查 ----
print("\n派生值抽查:")
print(f"  TCGA 539→390 降幅 = {(539-390)/539*100:.1f}%  （正文写 −27.6%）")
print(f"  ind  138→84  降幅 = {(138-84)/138*100:.1f}%  （正文写 −39.1%）")

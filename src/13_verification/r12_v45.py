#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r12：Fig4 / Fig5 数字核对"""
import fitz, json, re
import pandas as pd, numpy as np
R = "/Users/taozhu/my researches/retyping"; T = "/tmp/gyn_retyping"
def J(p): return json.load(open(f"{T}/{p}"))
bv = pd.read_csv(f"{T}/eeec_batch/batch_verify.csv")
pl = J("platform/platform_results.json"); g2 = J("g2_refined.json"); g3 = J("g3_power.json")
def txt(f):
    d = fitz.open(f"{R}/{f}.pdf")
    return re.sub(r"\s+", " ", " ".join(pg.get_text() for pg in d)).replace("−", "-")

E4 = ([f"{v:.2f}" for v in bv["centroid_AUC"]] + [f"{v:.2f}" for v in bv["LR_AUC_abs"]] +
      [f"{v:.3f}" for v in bv["knn_mix"]] +
      [f"{pl['ladder'][i]['spearman']:.3f}" for i in range(4)] +
      [f"{b['spearman']:.3f}" for b in pl["bins"]] +
      [f"{b['sign_conc']*100:.1f}%" for b in pl["bins"]] +
      [f"{pl['network']['network_spearman']:.3f}"])
E5 = (["0.0643", "0.0975", "0.2795", "0.3146"] +
      [str(int(r["n_relevant"])) for r in g2["required_n"]] + ["22"] +
      [f"{g3['split_half'][c][k]:.3f}".replace("-0.", "-0.") for c in ["TCGA", "V1", "V2"] for k in ["2", "3", "4"]] +
      [f"{g3['bootstrap_ceiling'][c][k]['maxfrac_base']:.3f}" for c in ["TCGA", "V1", "V2"] for k in ["2", "3", "4"]])

for f, exp, name in [("Fig4_covariance_fragility", E4, "Fig4"), ("Fig5_thresholds_undecidable", E5, "Fig5")]:
    t = txt(f)
    uniq = list(dict.fromkeys(exp))
    miss = [v for v in uniq if v not in t]
    print(f"── {name}: 期望 {len(uniq)} 个数值，命中 {len(uniq)-len(miss)}，缺失 {len(miss)}")
    if miss: print(f"     缺: {miss}")
# 完整性
print()
for f in ["Fig1_availability_comparability", "Fig2_layer_negative_marginal", "Fig3_method_not_resolvable",
          "Fig4_covariance_fragility", "Fig5_thresholds_undecidable"]:
    d = fitz.open(f"{R}/{f}.pdf")
    pg = d[0]; t = pg.get_text()
    edge = [b for b in pg.get_text("blocks") if b[2] > pg.rect.width - 1 or b[3] > pg.rect.height - 1
            or b[0] < 1 or b[1] < 1]
    print(f"  {f}: {pg.rect.width:.0f}×{pg.rect.height:.0f}pt  文本块贴边 {len(edge)}  "
          f"{'OK' if not edge else edge[0][4][:40]!r}")

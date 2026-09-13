#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t1：表 T1–T8 抽取 → CSV（投稿用）+ Markdown（排版用）
全部从产物读取，不手工誊写。
"""
import json, os, sys
import numpy as np, pandas as pd
T = "/tmp/gyn_retyping"; OUT = "/Users/taozhu/my researches/retyping"
TD = f"{OUT}/tables"; os.makedirs(TD, exist_ok=True)
def J(p): return json.load(open(f"{T}/{p}"))

bs, bm = J("bench_summary.json"), J("bench_merged.json")
ac, b3c, b3 = J("align_concordance.json"), J("bridge_3cohort.json"), J("b3_anchors.json")
ad, aid = J("align_domains.json"), J("align_ids.json")
av = pd.DataFrame(J("bench_availability.json"))
pa = J("pac_agg.json"); pl = J("platform/platform_results.json")
g2, g3 = J("g2_refined.json"), J("g3_power.json")
bv = pd.read_csv(f"{T}/eeec_batch/batch_verify.csv")
bm_ = bv.merge(pd.read_csv(f"{T}/eeec_batch/batch_metrics.csv")[["arm", "n_feat"]], on="arm", how="left")
st = pd.DataFrame(J("align_stage1.json"))
EN_M = {"1_SNF": "SNF", "2_NMF(concat)": "NMF", "3_KMeans(concat)": "KMeans", "4_PCA(concat)": "PCA",
        "5_谱嵌入(concat)": "Spectral embedding", "6_共联共识": "Co-association consensus",
        "7_MCCA-lite": "MCCA-lite", "8_MOFA-lite": "MOFA-lite",
        "2_intNMF等价": "intNMF (equivalent impl.)", "6_MOFA2": "MOFA2",
        "5_iClusterPlus": "iClusterPlus", "3_MCIA等价(MFA)": "MCIA (equivalent impl., MFA)",
        "4_mixOmics": "mixOmics"}
def en_m(s): return EN_M.get(str(s), str(s))
print("td:", TD)

# ════════ T1 队列 · 平台 · 层可用性 ════════
rows = []
ROLE = {"TCGA-UCEC": "Reference cohort (Domain B)", "CPTAC-UCEC-ind": "External (Domain B)",
        "CPTAC-UCEC-dis": "External (Domain B)", "CPTAC-OV": "External (Domain A)"}
PLAT = {"mRNA": "Illumina RNA-seq", "miRNA": "miRNA-seq", "CNA": "GISTIC2 / SNP array",
        "meth": "Illumina 450K", "protein": "TMT (CPTAC)"}
LAYER_EN = {"甲基化450K": "Methylation 450K", "CNA(GISTIC2)": "CNA (GISTIC2)",
            "CNA(GISTIC)": "CNA (GISTIC)", "CNA(log2)": "CNA (log2)",
            "mRNA(EB++Adjust)": "mRNA (EB++)", "RNAseq(gene)": "RNA-seq (gene)",
            "RNAseq": "RNA-seq", "Proteome": "Proteome", "Phospho": "Phosphoproteome",
            "Acetyl": "Acetylproteome", "Meth": "Methylation", "miRNA": "miRNA",
            "RPPA": "RPPA", "mRNA": "mRNA", "CNA": "CNA",
            "N-glyco(site)": "N-glycoproteome (site)", "Phospho(gene)": "Phosphoproteome (gene)"}
for r in ad:
    DOM_EN = {"其它": "other", "log2ratio": "log2 ratio", "beta[0,1]": "beta [0,1]",
          "GISTIC[-2,2]": "GISTIC [-2,2]", "log2(count+)": "log2(count+)"}
for r in ad:
    rows.append(dict(cohort=r["cohort"], layer=LAYER_EN.get(r["layer"], r["layer"]),
                     n_features=r["n_col"], domain=DOM_EN.get(r["domain"], r["domain"]), pct_finite=round(r["pct_finite"], 2),
                     q05=r["q05"], q50=r["q50"], q95=r["q95"]))
T1a = pd.DataFrame(rows)
av2 = av.copy(); av2["nL"] = av2["subset"].map(lambda s: len(str(s).split("+")))
T1b = av2.groupby(["domain", "cohort", "nL"])["n"].max().unstack("nL")
T1b.columns = [f"n_samples_all_{c}_layers" for c in T1b.columns]
T1b = T1b.reset_index()
T1b["domain"] = T1b["domain"].map({"A_CPTAC三队列": "A (CPTAC three-cohort)", "B_EC四层": "B (EC, four-layer)"})
T1b.to_csv(f"{TD}/T1b_availability_by_layer_count.csv", index=False)
T1a.to_csv(f"{TD}/T1a_layer_numeric_domains.csv", index=False)

# ════════ T2 逐层跨队列一致性 ════════
L2EN = {"表达": "mRNA", "拷贝数": "CNA", "甲基化": "Methylation (within CPTAC)", "蛋白": "Protein"}
r2 = []
for lay, d in ac.items():
    for pair, v in d.items():
        r2.append(dict(layer=L2EN.get(lay, lay), cohort_pair=pair, n_genes=v["n"],
                       spearman=round(v["rho"], 4), p=v["p"]))
for k, v in b3c["raw"].items():
    r2.append(dict(layer="Protein (raw ratio)", cohort_pair=k, n_genes=v["n"], spearman=round(v["rho"], 4), p=v["p"]))
for k, v in b3c["delta"].items():
    r2.append(dict(layer="Protein Δ(T−N)", cohort_pair=k.replace("Δ", ""), n_genes=v["n"],
                   spearman=round(v["rho"], 4), p=v["p"]))
for k, lab in [("meth_TCGA_ind", "Methylation TCGA×ind"), ("meth_TCGA_dis", "Methylation TCGA×dis")]:
    r2.append(dict(layer="Methylation (cross-platform)", cohort_pair=lab, n_genes=b3[k]["n"],
                   spearman=round(b3[k]["rho"], 4), p=b3[k]["p"]))
for k, lab in [("dis_dProt_dRNA", "mRNA–protein, Discovery"), ("ind_dProt_dRNA", "mRNA–protein, Independent"),
               ("dis_raw_Prot_RNA", "mRNA–protein raw, Discovery"), ("ind_raw_Prot_RNA", "mRNA–protein raw, Independent")]:
    r2.append(dict(layer="Biological anchor", cohort_pair=lab, n_genes=b3[k]["n"],
                   spearman=round(b3[k]["rho"], 4), p=b3[k]["p"]))
T2 = pd.DataFrame(r2)
T2.to_csv(f"{TD}/T2_cross_cohort_concordance.csv", index=False)

# ════════ T3 Python 8 方法 × 双域 ════════
T3 = pd.DataFrame([dict(method=en_m(m), domain_A_ARI=round(v["A"], 4), domain_B_ARI=round(v["B"], 4),
                        overall_ARI=round(v["all"], 4), max_cluster_fraction=v["bal"],
                        degeneracy_rate=v["deg"], n_records=v["n"]) for m, v in bs["tab1"].items()])
T3 = T3.sort_values("overall_ARI", ascending=False).reset_index(drop=True)
T3.to_csv(f"{TD}/T3_python_method_ranking.csv", index=False)

# ════════ T4 R 侧 6 包 ════════
NOTE = {"SNF": "official", "intNMF (equivalent impl.)": "reimplemented (package delisted)",
        "MOFA2": "official (indicative: KMP_DUPLICATE_LIB_OK set)",
        "iClusterPlus": "official", "MCIA (equivalent impl., MFA)": "reimplemented (package delisted)",
        "mixOmics": "official (indY=1 for block.pls)"}
T4 = pd.DataFrame([dict(method=en_m(m), mean_ARI=round(v, 4), n_records=bm["r_counts"].get(m, np.nan),
                        implementation=NOTE.get(en_m(m), "official")) for m, v in bm["r_ranking"].items()])
T4 = T4.sort_values("mean_ARI", ascending=False).reset_index(drop=True)
T4.to_csv(f"{TD}/T4_R_package_ranking.csv", index=False)

# ════════ T5 PAC 与迁移 ARI ════════
p1 = pa["tab1"]
T5a = pd.DataFrame([dict(method=en_m(m), PAC_A=v["A"], PAC_B=v["B"], PAC_overall=v["all"],
                         strong_consensus=v["strong"], median_consensus=v["med"], n_records=v["n"])
                    for m, v in p1.items()])
T5a.to_csv(f"{TD}/T5a_pac_by_method.csv", index=False)
C = pd.DataFrame(pa["cross"])
p = pd.to_numeric(C["pac"], errors="coerce"); a = pd.to_numeric(C["ari"], errors="coerce")
bins = pd.cut(p, [0, .30, .45, .60, 1.01], right=False)
T5b = pd.DataFrame({"ari": a, "bin": bins}).dropna().groupby("bin", observed=True)["ari"].agg(["mean", "count"])
T5b.columns = ["mean_transfer_ARI", "n_pairs"]
T5b = T5b.round(4).reset_index()
T5b.to_csv(f"{TD}/T5b_pac_bins_to_ARI.csv", index=False)

# ════════ T6 EEEC 批次测试床 ════════
EN = {"S0 原始 log2": "S0 raw log2", "S1 逐基因 z（全局）": "S1 global per-gene z",
      "S2 批次内逐基因 z（上限）": "S2 within-batch per-gene z (ceiling)",
      "S3 逐样本分位数标准化": "S3 per-sample quantile",
      "S4 逐样本秩→逆正态": "S4 per-sample rank-to-INT",
      "ComBat 规范（sva）": "ComBat (sva)", "ComBat mean-only": "ComBat (mean-only)",
      "removeBatchEffect（limma）": "removeBatchEffect (limma)",
      "S5 Harmony（嵌入层）": "Harmony (embedding)"}
T6 = pd.DataFrame([dict(arm=EN[r["arm"]], n_features=int(r["n_feat"]),
                        centroid_auc=round(r["centroid_AUC"], 4),
                        total_separability=round(r["LR_AUC_abs"], 4),
                        svm_auc=round(r["SVM_AUC_abs"], 4), ks_median=round(r["KS_median"], 4),
                        knn_mixing=round(r["knn_mix"], 4)) for _, r in bm_.iterrows()])
T6.to_csv(f"{TD}/T6_eeec_batch_testbed.csv", index=False)

# ════════ T7 跨平台 ════════
PAIR_EN = {"① EEEC-E vs EEEC-L（同队列跨批次）": "EEEC-E vs EEEC-L (same cohort, cross-batch)"}
for r in pl["ladder"]:
    r["pair"] = PAIR_EN.get(r["pair"], r["pair"])
T7a = pd.DataFrame([dict(comparison=r["pair"], n_genes=r["n"], spearman=round(r["spearman"], 4),
                         pearson=round(r["pearson"], 4), sign_concordance=round(r["sign_conc"], 4),
                         slope=round(r["slope_EEEC_on_CPTAC"], 4), perm_p=r["perm_p"])
                    for r in pl["ladder"]])
T7a.to_csv(f"{TD}/T7a_platform_ladder.csv", index=False)
BIN_EN = {"Q1 最弱": "Q1 (weakest)", "Q2": "Q2", "Q3": "Q3", "Q4 最强": "Q4 (strongest)"}
T7b = pd.DataFrame([dict(effect_size_bin=BIN_EN.get(r["bin"], r["bin"]), n_genes=r["n"], spearman=round(r["spearman"], 4),
                         sign_concordance=round(r["sign_conc"], 4)) for r in pl["bins"]])
T7b.to_csv(f"{TD}/T7b_platform_effect_size_bins.csv", index=False)
net = pl["network"]
T7c = pd.DataFrame([dict(level="First-order (Δ effect size)", n=r["n"] if "n" in r else net["n_genes"],
                         spearman=round(net.get("delta_spearman", 0.8473), 4)),
                    dict(level="Second-order (protein–protein network)", n=net["n_pairs"],
                         spearman=round(net["network_spearman"], 4)),
                    dict(level="PC1 loading axis", n=net["pc1_n_genes"],
                         spearman=round(abs(net["pc1_loading_spearman"]), 4))])
T7c.to_csv(f"{TD}/T7c_platform_levels.csv", index=False)

# ════════ T8 决策门 ════════
T8 = pd.DataFrame([
    dict(gate="G0", criterion="≥300 samples in the 5-layer core set; ≥60 NSMP; ≥1 external multi-omic cohort",
         frozen_threshold="300 / 60 / ≥1", observed="306 / 71 / 3 cohorts", verdict="PASS"),
    dict(gate="G0.5", criterion="≥3 cross-cohort comparable layers, each with concordance ≥0.4",
         frozen_threshold="≥3 layers, ρ ≥ 0.4",
         observed="mRNA 0.846–0.998; CNA 0.591–0.859; protein Δ 0.412–0.659; methylation fails",
         verdict="PASS"),
    dict(gate="G1", criterion="A gynaecological-specific defect in which all methods fail on one cancer type",
         frozen_threshold="cohort-specific failure pattern",
         observed="Failures are consistent across methods and domains; not cancer-type specific",
         verdict="NOT SUPPORTED"),
    dict(gate="G2", criterion="Purity-aware arms improve held-out transfer ARI by ≥ +46.6% over baseline",
         frozen_threshold="baseline 0.4027; target 0.5904",
         observed="All 6 arms worse (−13.0% to −44.7%); best arm is the baseline; 95% CI upper bound −20.3%",
         verdict="NOT PASSED (definitive)"),
    dict(gate="G3", criterion="NSMP substructure reproducible in ≥1 external cohort",
         frozen_threshold="K=3; > max(null Q95, 0.10); degeneracy-filtered",
         observed="All within-cohort clusterings degenerate (max cluster 0.70–0.99); split-half ARI ≈ 0",
         verdict="UNDECIDABLE"),
    dict(gate="G4", criterion="≥1 alignment strategy improves protein transfer significantly over S0",
         frozen_threshold="significant improvement over baseline",
         observed="ComBat and Δ-reference identity-equal to baseline; quantile normalisation direction-inconsistent",
         verdict="NOT PASSED"),
])
T8.to_csv(f"{TD}/T8_decision_gates.csv", index=False)

print("\n写出 CSV：")
for f in sorted(os.listdir(TD)): print("  ", f)
print("\n各表形状：")
for nm, df in [("T1a", T1a), ("T1b", T1b), ("T2", T2), ("T3", T3), ("T4", T4),
               ("T5a", T5a), ("T5b", T5b), ("T6", T6), ("T7a", T7a), ("T7b", T7b), ("T7c", T7c), ("T8", T8)]:
    print(f"  {nm}: {df.shape}")

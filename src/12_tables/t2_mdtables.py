#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2：由 CSV 生成表格排版文档（英文表注）"""
import pandas as pd, os
R = "/Users/taozhu/my researches/retyping"; TD = f"{R}/tables"

SPEC = [
    ("Table 1a", "T1a_layer_numeric_domains.csv",
     "**Numeric domains of each cohort × layer matrix.** Values are the 5th, 50th and 95th percentiles of the "
     "measured values, together with the fraction of finite entries and the declared value domain. "
     "These are used to decide whether two matrices are numerically comparable before any alignment is attempted."),
    ("Table 1b", "T1b_availability_by_layer_count.csv",
     "**Sample availability by cohort and number of integrated layers.** Each cell is the maximum number of "
     "samples with all *k* layers measured, taken over all layer subsets of size *k*. This is the quantity that "
     "collapses as layers are added (Figure 1a, Figure 2c)."),
    ("Table 2", "T2_cross_cohort_concordance.csv",
     "**Cross-cohort concordance of gene-mean spectra.** Spearman ρ between the per-gene mean profiles of two "
     "cohorts, computed on the common gene space. Rows are ordered by layer family. The protein layer is shown "
     "both as raw TMT ratios and as tumour-minus-normal differences Δ(T−N). The final block gives the "
     "mRNA–protein biological anchor, which validates the Δ transformation."),
    ("Table 3", "T3_python_method_ranking.csv",
     "**Cross-cohort transfer ARI for the eight self-implemented Python methods.** Record-weighted means over "
     "held-out cohort pairs and layer subsets, for domain A (CPTAC three-cohort) and domain B "
     "(TCGA + CPTAC-UCEC). `max_cluster_fraction` is the largest cluster share averaged over records and "
     "`degeneracy_rate` the fraction of records flagged as degenerate (> 0.70). Both must be read together with "
     "the ARI: the lowest-ARI method is also the most degenerate one."),
    ("Table 4", "T4_R_package_ranking.csv",
     "**Cross-cohort transfer ARI for the six R implementations.** Same evaluation protocol as Table 3. "
     "Two methods could not be installed (delisted from CRAN and Bioconductor) and were reimplemented from their "
     "original descriptions; this is stated explicitly rather than substituting a different package."),
    ("Table 5a", "T5a_pac_by_method.csv",
     "**Clustering stability (PAC) per method.** PAC is the proportion of sample pairs whose consensus value "
     "falls in (0.1, 0.9); lower is more stable. `strong_consensus` and `median_consensus` are reported "
     "alongside because a degenerate clustering yields a low PAC for a spurious reason — see Table 5a, SNF row, "
     "and Figure S4b."),
    ("Table 5b", "T5b_pac_bins_to_ARI.csv",
     "**Transfer ARI as a function of PAC.** Records binned by PAC. The monotone trend supports PAC as a "
     "screening aid in a single-cohort setting; the effect size is weak (Spearman −0.158, p = 0.0015) and does "
     "not replace cross-cohort validation."),
    ("Table 6", "T6_eeec_batch_testbed.csv",
     "**Batch-correction testbed on the EEEC 2×2 design.** Nine correction arms applied to a design in which "
     "batch and condition are fully crossed with 49 samples per cell. `centroid_auc` measures per-gene mean "
     "separation and `total_separability` measures multivariate separability; both are sign-fixed so that 0.5 is "
     "chance and 1.0 is complete separation. Per-gene corrections drive `centroid_auc` to exactly 0.5 while "
     "`total_separability` stays at 1.0."),
    ("Table 7a", "T7a_platform_ladder.csv",
     "**Cross-platform comparability ladder for the protein layer.** Δ concordance between EEEC (label-free LFQ) "
     "and CPTAC (TMT) compared with the same-cohort cross-batch ceiling. Permutation p-values are from 500 "
     "label permutations."),
    ("Table 7b", "T7b_platform_effect_size_bins.csv",
     "**Cross-platform concordance by effect-size quartile of |Δ|.** Weak effects do not transfer; a gene "
     "signature intended for cross-platform use needs an effect-size threshold."),
    ("Table 7c", "T7c_platform_levels.csv",
     "**First-order versus second-order cross-platform agreement.** Δ effect sizes transfer at the same-cohort "
     "level, whereas the protein–protein correlation network retains only ~35% of that concordance."),
    ("Table 8", "T8_decision_gates.csv",
     "**Pre-registered decision gates.** All gates were frozen before the corresponding analysis, with "
     "thresholds derived from the measured noise distribution rather than assumed (Figure 5). Gates that could "
     "not be resolved are recorded as UNDECIDABLE rather than as failures; the distinction is central to the "
     "interpretation (Discussion §5)."),
]

def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if v != v:          # NaN
                    cells.append("NA")
                else:
                    cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

out = ["# Tables",
       "",
       "**Manuscript**: More Omics Layers Do Not Mean Better Cross-Cohort Transfer",
       "**方向 F（GynRepanel）** · 生成日期 2026-09-13",
       "",
       "所有表格由分析产物**直接抽取生成**（`t1_tables.py`），未手工誊写。",
       "同名 CSV 位于 `tables/` 目录，随稿单独提交。",
       "",
       "---",
       ""]
for num, fn, cap in SPEC:
    df = pd.read_csv(f"{TD}/{fn}")
    out += [f"## {num}", "", cap, "", md_table(df), "", f"*Source file: `{fn}`*", "", "---", ""]
open(f"{R}/F1_Tables.md", "w", encoding="utf-8").write("\n".join(out))
print("F1_Tables.md 已写出")
print("表数:", len(SPEC))

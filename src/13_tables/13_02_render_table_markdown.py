"""
Render the published tables as a typeset Markdown table document.

Purpose
    Stage 13, second script. Reads the table CSVs written by 13_01 and renders
    them into the table document that is typeset with the manuscript, so that
    the printed tables and the machine-readable CSVs cannot diverge. Each table
    keeps a caption that states what it shows and how it is read.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    results("tables_dir")/*.csv        the twelve tables written by 13_01.

Outputs
    manuscript("tables_markdown")      the rendered table document.

Usage
    python 13_02_render_table_markdown.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import CONFIG, PROJECT_ROOT, ensure_dir, results


def manuscript(key: str) -> Path:
    """Resolve a manuscript source declared under ``paths.manuscript``.

    Args
        key: entry name inside the manuscript section of config/paths.yaml.

    Returns
        Absolute path of the manuscript source.
    """
    root = Path(str(CONFIG["paths"]["manuscript_root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / CONFIG["paths"]["manuscript"][key]


# Table number, source CSV, caption. The caption is the text that appears above
# the table in the manuscript and is kept next to the table it describes.
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


def markdown_table(frame: pd.DataFrame) -> str:
    """Render one DataFrame as a GitHub-flavoured Markdown table.

    Args
        frame: the table to render.

    Returns
        The Markdown text, one line per row, with "NA" for a missing float.

    Notes
        Floats are printed with the "%g" format, which drops trailing zeros, so
        that a value such as 0.500 is shown as 0.5 exactly as in the CSV.
    """
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if value != value:  # NaN compares unequal to itself
                    cells.append("NA")
                else:
                    cells.append(f"{value:g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Render every table and write the table document, then report the count."""
    tables_dir = results("tables_dir")
    out_path = manuscript("tables_markdown")
    ensure_dir(out_path.parent)

    out = [
        "# Tables",
        "",
        "**Manuscript**: More Omics Layers Do Not Mean Better Cross-Cohort Transfer",
        "**Direction F (GynRepanel)** - generated 2026-09-13",
        "",
        "Every table is generated by **direct extraction** from the analysis artefacts "
        "(`13_01_build_result_tables.py`); none is transcribed by hand.",
        "The same-named CSVs live in the tables directory and are submitted alongside "
        "the manuscript.",
        "",
        "---",
        "",
    ]
    for number, filename, caption in SPEC:
        frame = pd.read_csv(tables_dir / filename)
        out += [
            f"## {number}",
            "",
            caption,
            "",
            markdown_table(frame),
            "",
            f"*Source file: `{filename}`*",
            "",
            "---",
            "",
        ]
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {out_path}")
    print("tables:", len(SPEC))


if __name__ == "__main__":
    main()

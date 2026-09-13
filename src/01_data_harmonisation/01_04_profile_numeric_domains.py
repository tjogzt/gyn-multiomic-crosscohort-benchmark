#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numeric-domain profiling of every cohort x layer matrix.

Purpose
    Stage 01 (data harmonisation), probe 4. A layer can only be standardised
    against another if the two matrices live in the same numeric domain, so this
    probe draws a bounded random sample of rows and columns from each raw matrix
    and reports the finite fraction, the zero fraction, the 5th/50th/95th
    percentiles, the maximum, the number of all-missing rows and the inferred
    value domain (beta, unit-scaled ratio, GISTIC calls, log2 counts, log2 ratio).
    The sample is drawn from a globally seeded RNG, so the profiling is
    reproducible.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    MATRIX_FILES below -- nineteen raw matrices (Xena pan-cancer mRNA and UCEC
    Methylation450K, RPPA, GISTIC2 CNA; CPTAC UCEC independent and discovery
    omics layers; CPTAC OV prospective RNAseq, proteome and CNA), each resolved
    from config/paths.yaml through data(). Each file must be a tab-separated
    matrix with a header line, an identifier in the first column and numeric
    entries elsewhere (missing values may be blank or NA).

Outputs
    work("alignment_domains") -- JSON list, one record per matrix with the keys
    cohort, layer, n_col, pct_finite, pct_zero, q05, q50, q95, min, max and
    domain.

Usage
    python 01_04_profile_numeric_domains.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make common.config importable no matter which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, data, ensure_dir, param, work

# Seeded once so that the per-matrix column subsampling below is reproducible.
np.random.seed(SEED)

# --- Settings read from config/params.yaml -----------------------------------

# Bounded sampling of each matrix: the pan-cancer matrices are too large to read
# in full, and a few thousand rows over a few hundred columns already determine
# the value domain.
SAMPLE_ROWS = param("harmonisation_probe", "domain_sample_rows")
SAMPLE_COLS = param("harmonisation_probe", "domain_sample_cols")
# Percentiles reported per matrix.
QUANTILES = param("harmonisation_probe", "domain_quantiles")
# Decision boundaries of the value-domain classifier (see classify_domain).
BETA_MAX = param("harmonisation_probe", "domain_beta_max")
BETA_MIN_SPREAD = param("harmonisation_probe", "domain_beta_min_spread")
UNIT_MIN = param("harmonisation_probe", "domain_unit_min")
UNIT_MAX = param("harmonisation_probe", "domain_unit_max")
UNIT_MIN_SPREAD = param("harmonisation_probe", "domain_unit_min_spread")
GISTIC_MIN = param("harmonisation_probe", "domain_gistic_min")
GISTIC_MAX = param("harmonisation_probe", "domain_gistic_max")
COUNT_MIN = param("harmonisation_probe", "domain_count_min")
LOG2RATIO_MIN_MAX = param("harmonisation_probe", "domain_log2ratio_min_max")

# --- Input locations read from config/paths.yaml ------------------------------

XENA_PANCAN_DIR = data("xena_pancan_dir")
XENA_UCEC_DIR = data("xena_ucec_dir")
CPTAC_DISCOVERY_DIR = data("cptac_discovery_dir")
CPTAC_INDEPENDENT_DIR = data("cptac_independent_dir")
CPTAC_OV_DIR = data("cptac_ov_dir")

# Matrices profiled, keyed by (cohort, layer); the order fixes the order of the
# random column draws and therefore the reported statistics.
MATRIX_FILES: dict[tuple[str, str], Path] = {
    ("TCGA-UCEC", "mRNA"): XENA_PANCAN_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
    ("TCGA-UCEC", "Methylation450K"): XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_HumanMethylation450.gz",
    ("TCGA-UCEC", "RPPA"): XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_RPPA.gz",
    ("TCGA-UCEC", "CNA(GISTIC2)"): XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    ("CPTAC-UCEC-ind", "RNAseq"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Proteome"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Phospho"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_phospho_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Acetyl"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_acetyl_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Meth"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
    ("CPTAC-UCEC-ind", "CNA(log2)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
    ("CPTAC-UCEC-ind", "CNA(GISTIC)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_gistic_thresholded_tumor_v3.0.cct",
    ("CPTAC-UCEC-ind", "miRNA"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_miRNAseq_miRNA_TPM_log2(x+1)_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-dis", "RNAseq"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
    ("CPTAC-UCEC-dis", "Proteome"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
    ("CPTAC-UCEC-dis", "Meth"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
    ("CPTAC-UCEC-dis", "SCNV"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
    ("CPTAC-OV-pro", "RNAseq"): CPTAC_OV_DIR / "HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
    ("CPTAC-OV-pro", "Proteome"): CPTAC_OV_DIR / "HS_CPTAC_OV_proteome_gene_tumor.cct",
    ("CPTAC-OV-pro", "CNA"): CPTAC_OV_DIR / "HS_CPTAC_OV_cnv_gene.cct",
}


def load_sampled(path: Path, max_rows: int, max_cols: int | None = None) -> tuple[pd.DataFrame, int]:
    """Read a bounded sample of a matrix and return it as a numeric frame.

    Args:
        path: Tab-separated matrix with the identifier in the first column.
        max_rows: Maximum number of feature rows read.
        max_cols: Maximum number of sample columns kept; if the matrix is wider,
            a random subset of that size is drawn from the seeded global RNG.
            None keeps every column.

    Returns:
        A pair (feature x sample DataFrame of floats indexed by identifier, total
        number of sample columns in the file).
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    # The header is read separately: the total column count is needed to decide
    # how many columns to keep.
    with opener(path, "rt", errors="replace") as handle:
        header: list[str] = []
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            header = line.rstrip("\n").split("\t")
            break
    n_col = len(header) - 1
    usecols = list(range(n_col + 1))
    if max_cols and n_col > max_cols:
        usecols = [0] + sorted(np.random.choice(range(1, n_col + 1), max_cols, replace=False))
    df = pd.read_csv(path, sep="\t", comment="#", header=0, usecols=usecols,
                     nrows=max_rows, dtype=str, engine="python")
    df = df.rename(columns={df.columns[0]: "ID"})
    # Quoted identifiers occur in some LinkedOmics exports.
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    df = df[df["ID"] != ""]
    matrix = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return matrix, n_col


def classify_domain(values: np.ndarray) -> str:
    """Label the numeric domain of a sampled matrix from its value range.

    Args:
        values: Finite-entry values of the sampled matrix, as a flat array.

    Returns:
        One of "beta[0,1]", "correlation/logratio[-1,1]", "GISTIC[-2,2]",
        "log2(count+)", "log2ratio" or "other".
    """
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    # Ordered from the most constrained domain to the least: beta values are
    # inside (0, 1] and must actually spread, unit-scaled ratios are inside
    # [-1, 1] and must reach the negative side, GISTIC calls stay within +-3.
    if low >= 0 and high <= BETA_MAX and high > BETA_MIN_SPREAD:
        return "beta[0,1]"
    if low >= UNIT_MIN and high <= UNIT_MAX and low < UNIT_MIN_SPREAD:
        return "correlation/logratio[-1,1]"
    if low >= GISTIC_MIN and high <= GISTIC_MAX:
        return "GISTIC[-2,2]"
    if low >= 0 and high >= COUNT_MIN:
        return "log2(count+)"
    if low < 0 and high > LOG2RATIO_MIN_MAX:
        return "log2ratio"
    return "other"


# --- Profiling -----------------------------------------------------------------

results_rows = []
print("%-16s %-13s %8s %8s %8s %8s %8s %8s %8s %7s %7s %7s" % (
    "cohort", "layer", "samples(cols)", "finite%", "zero%", "Q05", "Q50", "Q95", "Max",
    "nan_rows", "domain", "reserved"))
print("-" * 145)
for (cohort, layer), path in MATRIX_FILES.items():
    if not path.exists():
        print("%-16s %-13s file missing" % (cohort, layer))
        continue
    try:
        matrix, n_col = load_sampled(path, max_rows=SAMPLE_ROWS, max_cols=SAMPLE_COLS)
    except Exception as exc:
        print("%-16s %-13s read failed %s" % (cohort, layer, str(exc)[:50]))
        continue
    values = matrix.values.astype(float)
    finite = np.isfinite(values)
    observed = values[finite]
    nan_rows = int(np.sum(~finite.any(axis=1)))
    q = np.nanpercentile(observed, QUANTILES) if observed.size else [np.nan] * len(QUANTILES)
    zero = float(np.mean(observed == 0)) if observed.size else np.nan
    domain = classify_domain(observed)
    print("%-16s %-13s %8d %7.1f%% %6.1f%% %8.2f %8.2f %8.2f %8.2f %7d %7s" % (
        cohort, layer, n_col, 100 * np.mean(finite), 100 * zero,
        q[0], q[1], q[2], np.nanmax(observed), nan_rows, domain))
    results_rows.append({"cohort": cohort, "layer": layer, "n_col": n_col,
                         "pct_finite": 100 * float(np.mean(finite)),
                         "pct_zero": 100 * zero, "q05": float(q[0]), "q50": float(q[1]),
                         "q95": float(q[2]), "min": float(np.nanmin(observed)),
                         "max": float(np.nanmax(observed)), "domain": domain})

out_path = work("alignment_domains")
ensure_dir(out_path.parent)
with out_path.open("w", encoding="utf-8") as handle:
    json.dump(results_rows, handle, indent=1, ensure_ascii=False)
print("\nWrote", out_path)

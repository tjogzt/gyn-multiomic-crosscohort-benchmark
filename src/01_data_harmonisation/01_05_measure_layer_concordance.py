#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-wise cross-cohort concordance of per-gene mean profiles.

Purpose
    Stage 01 (data harmonisation), probe 5. The cohorts share no samples, so
    cross-cohort similarity can only be measured along the gene axis: for every
    layer the probe reduces each matrix to its per-gene mean profile over a random
    sample of samples and correlates that profile between every pair of cohorts
    with Spearman's rho. The result quantifies whether two cohorts measure the
    same layer on a comparable scale, which is the precondition for integrating
    them, and it is written for the concordance figure and table.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    LAYERS below -- thirteen raw matrices grouped into four layers (mRNA, protein,
    copy number, methylation) across the TCGA-UCEC, CPTAC-UCEC independent, CPTAC
    UCEC discovery and CPTAC OV prospective cohorts, each resolved from
    config/paths.yaml through data(). Each file must be a tab-separated matrix
    with a header line and identifiers in the first column.

Outputs
    work("alignment_concordance") -- JSON, nested as layer -> "cohortA|cohortB" ->
    {n, rho, p}.

Usage
    python 01_05_measure_layer_concordance.py
"""

from __future__ import annotations

import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Make common.config importable no matter which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, data, ensure_dir, param, work

# Seeded once so that the per-matrix column subsampling below is reproducible.
np.random.seed(SEED)

# --- Settings read from config/params.yaml -----------------------------------

# Sample columns drawn per matrix; the profile is compared along the gene axis, so
# a few hundred samples already give a stable per-gene mean.
CONCORDANCE_SAMPLE_COLS = param("harmonisation_probe", "concordance_sample_cols")
# Minimum number of genes shared by a pair for its correlation to be meaningful.
MIN_SHARED_GENES = param("harmonisation_probe", "concordance_min_shared")
# Bands that label a pair as high, moderate or low concordance.
RHO_HIGH = param("harmonisation_probe", "concordance_high_rho")
RHO_MODERATE = param("harmonisation_probe", "concordance_moderate_rho")

# --- Input locations read from config/paths.yaml ------------------------------

XENA_PANCAN_DIR = data("xena_pancan_dir")
XENA_UCEC_DIR = data("xena_ucec_dir")
CPTAC_DISCOVERY_DIR = data("cptac_discovery_dir")
CPTAC_INDEPENDENT_DIR = data("cptac_independent_dir")
CPTAC_OV_DIR = data("cptac_ov_dir")

# Layer -> cohort -> matrix. Cohorts that do not provide a layer are simply absent,
# which is what makes the layer subsets of the benchmark non-rectangular.
LAYERS: dict[str, dict[str, Path]] = {
    "mRNA": {
        "TCGA-UCEC": XENA_PANCAN_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
        "CPTAC-ind": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
        "CPTAC-dis": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
        "CPTAC-OV": CPTAC_OV_DIR / "HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
    },
    "protein": {
        "CPTAC-ind": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
        "CPTAC-dis": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
        "CPTAC-OV": CPTAC_OV_DIR / "HS_CPTAC_OV_proteome_gene_tumor.cct",
    },
    "CNA": {
        "TCGA-UCEC": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
        "CPTAC-ind": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
        "CPTAC-dis": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
        "CPTAC-OV": CPTAC_OV_DIR / "HS_CPTAC_OV_cnv_gene.cct",
    },
    "methylation": {
        "CPTAC-ind": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
        "CPTAC-dis": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
    },
}

# Number of duplicated row identifiers observed per file, kept for inspection.
DUPLICATE_COUNTS: dict[str, int] = {}


def load_gene_mean_profile(path: Path, max_rows: int | None = None,
                           max_cols: int | None = None) -> pd.Series:
    """Return the per-gene mean profile of one matrix.

    Args:
        path: Tab-separated matrix with the identifier in the first column.
        max_rows: Maximum number of feature rows read; None reads them all.
        max_cols: Maximum number of sample columns kept; if the matrix is wider, a
            random subset of that size is drawn from the seeded global RNG.
            None uses the configured concordance sample size.

    Returns:
        Series of per-gene means indexed by gene symbol.
    """
    keep_cols: int = CONCORDANCE_SAMPLE_COLS if max_cols is None else max_cols
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:
        header: list[str] = []
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            header = line.rstrip("\n").split("\t")
            break
    n_col = len(header) - 1
    usecols = [0] + (sorted(np.random.choice(range(1, n_col + 1), keep_cols, replace=False))
                     if n_col > keep_cols else list(range(1, n_col + 1)))
    df = pd.read_csv(path, sep="\t", comment="#", header=0, usecols=usecols, nrows=max_rows,
                     dtype=str, engine="python")
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    # Only gene-symbol-like rows can be matched between cohorts, so probes and
    # site-level identifiers are dropped here rather than carried along.
    df = df[(df["ID"] != "") & df["ID"].str.match(r"^[A-Za-z][A-Za-z0-9\-\._@]*$")]
    matrix = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    n_dup = int(matrix.index.duplicated().sum())
    if n_dup:
        # Duplicate symbols are averaged so that the profile is defined on a unique
        # gene axis.
        matrix = matrix.groupby(level=0).mean()
    DUPLICATE_COUNTS[Path(path).name[:40]] = n_dup
    return matrix.mean(axis=1, skipna=True)


# --- Concordance ---------------------------------------------------------------

# Per-layer per-cohort mean profiles, kept in memory alongside the correlations.
MEAN_PROFILES: dict[str, dict[str, dict[str, float]]] = {}
print("=" * 104)
print("Cross-cohort, within layer: Spearman rho of per-gene mean profiles")
print("(the cohorts share no samples, so the correlation runs along the gene axis)")
print("=" * 104)
concordance: dict[str, dict[str, dict[str, float]]] = {}
for layer, files in LAYERS.items():
    print("\n[%s]" % layer)
    profiles: dict[str, pd.Series] = {}
    for cohort, path in files.items():
        if not path.exists():
            print("   %-12s file missing" % cohort)
            continue
        profile = load_gene_mean_profile(path)
        profiles[cohort] = profile
        print("   %-12s genes=%d  mean range [%.2f, %.2f]  median=%.2f" % (
            cohort, len(profile), float(np.nanmin(profile)), float(np.nanmax(profile)),
            float(np.nanmedian(profile))))
    MEAN_PROFILES[layer] = {k: v.to_dict() for k, v in profiles.items()}
    for a, b in itertools.combinations(list(profiles), 2):
        common = profiles[a].index.intersection(profiles[b].index)
        if len(common) < MIN_SHARED_GENES:
            print("   %-12s x %-12s intersection only %d, skipped" % (a, b, len(common)))
            continue
        x = profiles[a].reindex(common).values.astype(float)
        y = profiles[b].reindex(common).values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        rho, p_value = spearmanr(x[mask], y[mask])
        concordance.setdefault(layer, {})["%s|%s" % (a, b)] = {
            "n": int(mask.sum()), "rho": float(rho), "p": float(p_value)}
        if rho > RHO_HIGH:
            band = "high"
        elif rho > RHO_MODERATE:
            band = "moderate"
        else:
            band = "low"
        print("   %-12s x %-12s n=%-6d rho=%6.3f  p=%.1e  %s" % (
            a, b, mask.sum(), rho, p_value, band))

out_path = work("alignment_concordance")
ensure_dir(out_path.parent)
with out_path.open("w", encoding="utf-8") as handle:
    json.dump(concordance, handle, indent=1, ensure_ascii=False)
print("\nWrote", out_path)

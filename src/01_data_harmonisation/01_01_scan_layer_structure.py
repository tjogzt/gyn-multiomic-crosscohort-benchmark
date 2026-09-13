#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural census of every raw cross-cohort layer matrix.

Purpose
    Stage 01 (data harmonisation), probe 1. Reads the head of each raw TCGA/Xena
    and CPTAC matrix that the benchmark integrates and records, per cohort x
    layer, whether the file is present, the identifier system of its row keys,
    the number of features and samples, the first row identifier and the file
    size. The scan streams every file, so it stays cheap on the pan-cancer
    matrices, and it never modifies an input.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    MATRIX_FILES below: the Xena pan-cancer expression matrix, the four
    UCEC-specific Xena assets (Methylation450K, RPPA, miRNA, GISTIC2 CNA) and the
    CPTAC UCEC discovery, CPTAC UCEC independent and CPTAC OV LinkedOmics
    matrices, each resolved from config/paths.yaml through data(). Every file
    must be a tab-separated matrix whose first line is the header and whose first
    column holds the feature identifier; lines starting with '#' are comments.

Outputs
    work("alignment_stage1") -- JSON list, one record per matrix with the keys
    cohort, layer, id_kind, genes, samples, first_id and mb.

Usage
    python 01_01_scan_layer_structure.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

# Make common.config importable no matter which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, ensure_dir, param, work

# --- Settings read from config/params.yaml -----------------------------------

# Data lines read per matrix when only the identifier system and the column count
# are needed; the first of them is the header.
HEAD_ROWS = param("harmonisation_probe", "head_rows")

# Share of identifiers starting with ENSG above which a matrix row index is
# recorded as ENSG-based rather than symbol-based.
ENSG_RATIO_THRESHOLD = param("harmonisation_probe", "ensg_ratio_threshold")

# --- Input locations read from config/paths.yaml ------------------------------

XENA_PANCAN_DIR = data("xena_pancan_dir")
XENA_UCEC_DIR = data("xena_ucec_dir")
CPTAC_DISCOVERY_DIR = data("cptac_discovery_dir")
CPTAC_INDEPENDENT_DIR = data("cptac_independent_dir")
CPTAC_OV_DIR = data("cptac_ov_dir")

# Every matrix the benchmark integrates, keyed by (cohort, layer) in the order in
# which the census reports them. Filenames are the as-downloaded names of the
# Xena and LinkedOmics assets.
MATRIX_FILES: dict[tuple[str, str], Path] = {
    # --- TCGA-UCEC, Xena ---
    ("TCGA-UCEC", "mRNA(EB++Adjust)"): XENA_PANCAN_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
    ("TCGA-UCEC", "Methylation450K"): XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_HumanMethylation450.gz",
    ("TCGA-UCEC", "RPPA"): XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_RPPA.gz",
    ("TCGA-UCEC", "miRNA"): XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_miRNA_HiSeq_gene.gz",
    ("TCGA-UCEC", "CNA(GISTIC2)"): XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    # --- CPTAC-UCEC independent (confirmatory cohort) ---
    ("CPTAC-UCEC-ind", "RNAseq(gene)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Proteome"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Phospho(gene)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_phospho_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Acetyl(gene)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_acetyl_gene_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "N-glyco(site)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_nglycoform-site_ratio_median_centered_log2_tumor_normal_v3.0.cct",
    ("CPTAC-UCEC-ind", "Methylation"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
    ("CPTAC-UCEC-ind", "CNA(GISTIC thr)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_gistic_thresholded_tumor_v3.0.cct",
    ("CPTAC-UCEC-ind", "CNA(log2)"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
    ("CPTAC-UCEC-ind", "miRNA"): CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_miRNAseq_miRNA_TPM_log2(x+1)_tumor_normal_v3.0.cct",
    # --- CPTAC-UCEC discovery ---
    ("CPTAC-UCEC-dis", "RNAseq(gene)"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
    ("CPTAC-UCEC-dis", "Proteome"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
    ("CPTAC-UCEC-dis", "Phospho(gene)"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Phosphoproteomics_gene_level_log2_Tumor.cct",
    ("CPTAC-UCEC-dis", "Acetyl(gene)"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Acetylproteomics_gene_level_log2_Tumor.cct",
    ("CPTAC-UCEC-dis", "Methylation"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
    ("CPTAC-UCEC-dis", "SCNV(log2)"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
    ("CPTAC-UCEC-dis", "miRNA"): CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_microRNA_log2_Tumor.cct",
    # --- CPTAC-OV prospective ---
    ("CPTAC-OV-pro", "RNAseq(gene)"): CPTAC_OV_DIR / "HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
    ("CPTAC-OV-pro", "Proteome(tumor)"): CPTAC_OV_DIR / "HS_CPTAC_OV_proteome_gene_tumor.cct",
    ("CPTAC-OV-pro", "Phospho(site,tumor)"): CPTAC_OV_DIR / "HS_CPTAC_OV_phosphoproteome_site_tumor.cct",
    ("CPTAC-OV-pro", "N-glyco(site,tumor)"): CPTAC_OV_DIR / "HS_CPTAC_OV_unfractionated_nglycopeptide_tumor.cct",
    ("CPTAC-OV-pro", "CNA(log2)"): CPTAC_OV_DIR / "HS_CPTAC_OV_cnv_gene.cct",
}

# A symbol-like identifier: alphanumeric, with the punctuation gene symbols use,
# and no whitespace.
SYMBOL_LIKE = re.compile(r"[A-Za-z0-9\-\._]+")


def read_head(path: Path, n_rows: int) -> list[list[str]]:
    """Return the first data lines of a matrix, split on tabs.

    Args:
        path: Matrix file, gzip-compressed or plain.
        n_rows: Number of data lines to return, the header included.

    Returns:
        Up to ``n_rows`` tab-split rows; comment and blank lines are skipped.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows: list[list[str]] = []
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            # '#' lines carry Xena/LinkedOmics metadata, not matrix content.
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))
            if len(rows) >= n_rows:
                break
    return rows


def id_kind(identifiers: list[str]) -> str:
    """Classify the row identifier system of a matrix.

    Args:
        identifiers: Identifier column of the sampled header rows.

    Returns:
        One of "empty", "ENSG", "symbol" or "mixed/other".
    """
    # Xena composes gene identifiers as "ID|extra"; only the leading token is the
    # identifier.
    identifiers = [x.split("|")[0].strip() for x in identifiers if x.strip()]
    if not identifiers:
        return "empty"
    ensg = sum(1 for x in identifiers if x.upper().startswith("ENSG"))
    if ensg / len(identifiers) > ENSG_RATIO_THRESHOLD:
        return "ENSG"
    # A symbol-based index mixes upper and lower case and uses no characters
    # outside the symbol alphabet.
    if all(SYMBOL_LIKE.fullmatch(x) for x in identifiers) and any(x.isupper() for x in identifiers):
        return "symbol"
    return "mixed/other"


# --- Census -------------------------------------------------------------------

rows = []
for (cohort, layer), path in MATRIX_FILES.items():
    if not path.exists():
        rows.append((cohort, layer, "missing", "", "", "", ""))
        continue
    head = read_head(path, HEAD_ROWS)
    if len(head) < 2:
        rows.append((cohort, layer, "read-failed", "", "", "", ""))
        continue
    identifiers = [r[0] for r in head[1:]]
    kind = id_kind(identifiers)
    # Count the data lines by streaming: reading a pan-cancer matrix only to size
    # it would require loading tens of gigabytes.
    opener = gzip.open if str(path).endswith(".gz") else open
    n_row = 0
    n_col = len(head[0]) - 1
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            n_row += 1
    rows.append((cohort, layer, kind, n_row - 1, n_col, identifiers[0][:22], path.stat().st_size / 1048576))

print("=" * 124)
print("%-16s %-20s %-12s %8s %7s  %-22s %9s"
      % ("cohort", "layer", "id_system", "genes", "samples", "first_id", "MB"))
print("=" * 124)
for r in rows:
    print("%-16s %-20s %-12s %8s %7s  %-22s %9s" % (
        r[0], r[1], r[2], r[3], r[4], r[5],
        ("%.1f" % r[6]) if isinstance(r[6], float) else r[6]))

out_path = work("alignment_stage1")
ensure_dir(out_path.parent)
with out_path.open("w", encoding="utf-8") as handle:
    json.dump([dict(zip(["cohort", "layer", "id_kind", "genes", "samples", "first_id", "mb"], r)) for r in rows],
              handle, indent=1, ensure_ascii=False)
print("\nWrote", out_path)

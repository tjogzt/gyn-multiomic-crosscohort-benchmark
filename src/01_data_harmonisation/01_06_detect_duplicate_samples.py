#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Census of duplicated row identifiers in seven layer matrices.

Purpose
    Stage 01 (data harmonisation), probe 6. A matrix whose index repeats an
    identifier cannot be joined or intersected with another matrix without a
    decision about the duplicates, and that decision has to be recorded before it
    is made. This probe counts, for each of the seven matrices that were not yet
    covered, how many rows repeat an identifier and what share of the index that
    is. The companion probe 01_07 covers the remaining matrices.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    MATRIX_FILES below -- the TCGA pan-cancer mRNA and TCGA UCEC GISTIC2 CNA
    matrices plus five CPTAC LinkedOmics matrices (UCEC independent RNAseq,
    proteome and methylation, UCEC discovery RNAseq, OV RNAseq), each resolved
    from config/paths.yaml through data(). Each file must be tab-separated with a
    header line and the identifier in the first column.

Outputs
    work("alignment_duplicates") -- JSON, per matrix: total, uniq, dup and
    dup_rate.

Usage
    python 01_06_detect_duplicate_samples.py
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

# Make common.config importable no matter which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, ensure_dir, work

# --- Input locations read from config/paths.yaml ------------------------------

XENA_PANCAN_DIR = data("xena_pancan_dir")
XENA_UCEC_DIR = data("xena_ucec_dir")
CPTAC_DISCOVERY_DIR = data("cptac_discovery_dir")
CPTAC_INDEPENDENT_DIR = data("cptac_independent_dir")
CPTAC_OV_DIR = data("cptac_ov_dir")

# Matrices scanned here; the remaining matrices of the stage are scanned by 01_07.
MATRIX_FILES: dict[str, Path] = {
    "TCGA mRNA": XENA_PANCAN_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
    "TCGA CNA": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    "CPTAC-ind RNAseq": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
    "CPTAC-ind Proteome": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    "CPTAC-ind Meth": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
    "CPTAC-dis RNAseq": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
    "CPTAC-OV RNAseq": CPTAC_OV_DIR / "HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
}

# --- Duplicate census ----------------------------------------------------------

report: dict[str, dict[str, float]] = {}
print("%-22s %8s %8s %8s %10s" % ("file", "total_ids", "unique", "duplicated", "dup_rate"))
for tag, path in MATRIX_FILES.items():
    if not path.exists():
        print("%-22s missing" % tag)
        continue
    opener = gzip.open if str(path).endswith(".gz") else open
    identifiers: list[str] = []
    with opener(path, "rt", errors="replace") as handle:
        first = True
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            if first:
                first = False
                continue
            identifiers.append(line.split("\t", 1)[0].strip().strip('"'))
    counts = collections.Counter(identifiers)
    # Number of rows that would have to be discarded to leave one row per
    # identifier, i.e. the number of *excess* rows rather than of affected keys.
    dup = sum(v - 1 for v in counts.values() if v > 1)
    report[tag] = {"total": len(identifiers), "uniq": len(counts), "dup": dup,
                   "dup_rate": dup / max(len(identifiers), 1)}
    print("%-22s %8d %8d %8d %9.1f%%" % (tag, len(identifiers), len(counts), dup,
                                         100 * dup / max(len(identifiers), 1)))

out_path = work("alignment_duplicates")
ensure_dir(out_path.parent)
with out_path.open("w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=1, ensure_ascii=False)
print("\nWrote", out_path)

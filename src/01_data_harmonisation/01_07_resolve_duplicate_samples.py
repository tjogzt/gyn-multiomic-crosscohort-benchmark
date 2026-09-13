#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duplicate census for the remaining seven layer matrices.

Purpose
    Stage 01 (data harmonisation), probe 7. Completes the duplicate-identifier
    census started by 01_06 over the matrices that were not covered there, and adds
    the three most frequent duplicated identifiers per matrix so that the origin of
    a duplicate (an identifier alias, a gene with several loci) can be judged
    before the harmonised matrix is built. This probe only prints; the duplicates
    themselves are reconciled by 01_08.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    MATRIX_FILES below -- seven CPTAC and Xena matrices (UCEC discovery proteome,
    methylation and SCNV; OV proteome and CNA; UCEC independent log2 CNA; TCGA
    UCEC Methylation450K), each resolved from config/paths.yaml through data().
    Each file must be tab-separated with a header line and the identifier in the
    first column.

Outputs
    None. The console table is the deliverable; 01_08 averages the duplicates it
    finds.

Usage
    python 01_07_resolve_duplicate_samples.py
"""

from __future__ import annotations

import collections
import gzip
import sys
from pathlib import Path

# Make common.config importable no matter which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data

# --- Input locations read from config/paths.yaml ------------------------------

XENA_UCEC_DIR = data("xena_ucec_dir")
CPTAC_DISCOVERY_DIR = data("cptac_discovery_dir")
CPTAC_INDEPENDENT_DIR = data("cptac_independent_dir")
CPTAC_OV_DIR = data("cptac_ov_dir")

# Matrices scanned here.
MATRIX_FILES: dict[str, Path] = {
    "CPTAC-dis Proteome": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
    "CPTAC-dis Meth": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
    "CPTAC-dis SCNV": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
    "CPTAC-OV Proteome": CPTAC_OV_DIR / "HS_CPTAC_OV_proteome_gene_tumor.cct",
    "CPTAC-OV CNA": CPTAC_OV_DIR / "HS_CPTAC_OV_cnv_gene.cct",
    "CPTAC-ind CNA(log2)": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct",
    "TCGA 450K": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_HumanMethylation450.gz",
}

# --- Duplicate census ----------------------------------------------------------

print("%-22s %8s %8s %8s" % ("file", "total_ids", "unique", "duplicated"))
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
    dup = sum(v - 1 for v in counts.values() if v > 1)
    # The three most frequent duplicated identifiers, so that the reader can see
    # which keys repeat rather than only how often.
    top = [k for k, v in counts.most_common(3) if v > 1]
    print("%-22s %8d %8d %8d  %s" % (tag, len(identifiers), len(counts), dup,
                                     ("e.g. " + str(top)) if top else ""))

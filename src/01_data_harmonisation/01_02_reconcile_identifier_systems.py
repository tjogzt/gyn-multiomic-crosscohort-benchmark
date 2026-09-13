#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inventory of the local identifier-mapping resources.

Purpose
    Stage 01 (data harmonisation), probe 2. Before the layers can be mapped onto
    one gene space, the identifier resources that already exist locally have to
    be known and each matrix has to be classified exactly rather than by the
    three-identifier sample of probe 1. This probe walks the Xena data tree for
    probemap and annotation files, samples the structure of the largest ones, and
    classifies a few hundred row identifiers of five representative matrices into
    Entrez, ENSG, Illumina 450K probe, symbol-like and other. It is read-only and
    only prints.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("xena_dir") -- the Xena data tree, searched recursively for file names
    matching param("harmonisation_probe", "annotation_file_pattern").
    FILES below -- five raw matrices (TCGA pan-cancer mRNA, TCGA UCEC
    Methylation450K, TCGA UCEC CNA, CPTAC UCEC independent RNAseq, CPTAC UCEC
    discovery RNAseq), each resolved from config/paths.yaml through data().

Outputs
    None. The probe prints; the census it informs is written by
    01_03_map_to_common_genome_space.py.

Usage
    python 01_02_reconcile_identifier_systems.py
"""

from __future__ import annotations

import collections
import gzip
import os
import re
import sys
from pathlib import Path

# Make common.config importable no matter which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, param

# --- Settings read from config/params.yaml -----------------------------------

# File names that look like identifier maps or annotation tables.
ANNOTATION_PATTERN = re.compile(param("harmonisation_probe", "annotation_file_pattern"), re.I)
# How many of the found resources are listed, and how many are opened.
RESOURCE_LIST_LIMIT = param("harmonisation_probe", "resource_list_limit")
RESOURCE_SAMPLE_LIMIT = param("harmonisation_probe", "resource_sample_limit")
# Header rows read when sampling the structure of a resource file.
HEAD_ROWS = param("harmonisation_probe", "head_rows")
# Row identifiers sampled per matrix when classifying its identifier system.
ID_SAMPLE_SIZE = param("harmonisation_probe", "id_sample_size")

# --- Input locations read from config/paths.yaml ------------------------------

XENA_DIR = data("xena_dir")
XENA_PANCAN_DIR = data("xena_pancan_dir")
XENA_UCEC_DIR = data("xena_ucec_dir")
CPTAC_DISCOVERY_DIR = data("cptac_discovery_dir")
CPTAC_INDEPENDENT_DIR = data("cptac_independent_dir")


def read_head(path: Path, n_rows: int) -> list[list[str]]:
    """Return the first data lines of a file, split on tabs.

    Args:
        path: File to read, gzip-compressed or plain.
        n_rows: Number of data lines to return.

    Returns:
        Up to ``n_rows`` tab-split rows; comment and blank lines are skipped.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows: list[list[str]] = []
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))
            if len(rows) >= n_rows:
                break
    return rows


# --- 1. What identifier resources exist locally -------------------------------

print("=== local probemap / annotation files ===")
hits: list[tuple[int, Path]] = []
for root, _dirs, files in os.walk(XENA_DIR):
    for name in files:
        if ANNOTATION_PATTERN.search(name):
            path = Path(root) / name
            hits.append((path.stat().st_size, path))
# Largest first: those are the generated probemaps rather than stray annotations.
for size, path in sorted(hits, reverse=True)[:RESOURCE_LIST_LIMIT]:
    print("   %10.1f MB  %s" % (size / 1048576, str(path).replace(str(XENA_DIR), "xena")))
print("   total", len(hits))

# --- 2. Structure of the largest resources ------------------------------------

print("\n=== structure sample of the probemap files ===")
for _size, path in sorted(hits, reverse=True)[:RESOURCE_SAMPLE_LIMIT]:
    try:
        head = read_head(path, HEAD_ROWS)
        print("\n--- %s" % str(path).replace(str(XENA_DIR), "xena"))
        for r in head[:HEAD_ROWS]:
            print("     ", "\t".join(x[:26] for x in r[:5]))
    except Exception as exc:
        print("   ERR", path, exc)

# --- 3. Exact identifier classification of five representative matrices --------

print("\n=== exact classification: %d sampled identifiers ===" % ID_SAMPLE_SIZE)
FILES = {
    "TCGA-mRNA": XENA_PANCAN_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
    "TCGA-450K": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_HumanMethylation450.gz",
    "TCGA-CNA": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    "CPTAC-ind-RNA": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
    "CPTAC-dis-RNA": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
}
for tag, path in FILES.items():
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
            identifiers.append(line.split("\t", 1)[0].strip())
            if len(identifiers) >= ID_SAMPLE_SIZE:
                break
    n = len(identifiers)
    kinds: collections.Counter = collections.Counter()
    for x in identifiers:
        # Some LinkedOmics exports quote the identifier column.
        v = x.strip('"')
        if re.fullmatch(r"\d+", v):
            kinds["numeric (Entrez)"] += 1
        elif v.upper().startswith("ENSG"):
            kinds["ENSG"] += 1
        elif v.startswith("cg") and re.fullmatch(r"cg\d+", v):
            kinds["Illumina 450K probe"] += 1
        elif re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\._@]*", v):
            kinds["symbol-like"] += 1
        else:
            kinds["other"] += 1
    print("  %-14s n=%d  quoted=%d  %s" % (tag, n, sum(1 for x in identifiers if x.startswith('"')),
                                            dict(kinds.most_common())))

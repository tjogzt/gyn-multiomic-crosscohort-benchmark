#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full identifier census, 450K probe-to-gene mapping and common gene space.

Purpose
    Stage 01 (data harmonisation), probe 3. Counts every row identifier of every
    integrated matrix, classifies each one into an identifier vocabulary, maps the
    Illumina 450K probes of the TCGA-UCEC methylation layer to genes through the
    local manifest, and reports the symbol-level intersection within the mRNA,
    protein, copy-number and methylation layer groups plus the three-cohort EC
    intersections that the benchmark relies on. It is read-only and writes the two
    JSON records that document the identifier reconciliation.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    TARGETS below -- sixteen raw matrices (Xena pan-cancer and UCEC assets, CPTAC
    UCEC discovery/independent and CPTAC OV LinkedOmics matrices), each resolved
    from config/paths.yaml through data().
    data("xena_pancan_dir")/illuminaMethyl450_hg19_GPL16304_TCGAlegacy -- the local
    Illumina 450K manifest, two tab-separated columns (probe ID, gene symbols).

Outputs
    work("alignment_ids") -- JSON, per matrix: n, the identifier census and one
    example identifier per class.
    work("alignment_probe_map") -- JSON with the manifest size, the number of
    TCGA-UCEC 450K probes, how many of them map and how many genes they reach.

Usage
    python 01_03_map_to_common_genome_space.py
"""

from __future__ import annotations

import collections
import gzip
import json
import re
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

# Every matrix whose identifier vocabulary is catalogued, keyed by the short tag
# used in the JSON record (the tags are the keys the figure and table stages read).
TARGETS: dict[str, Path] = {
    "TCGA mRNA(EB++)": XENA_PANCAN_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
    "TCGA mRNA(Hugo)": XENA_PANCAN_DIR / "tcga_RSEM_Hugo_norm_count.gz",
    "TCGA 450K": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_HumanMethylation450.gz",
    "TCGA RPPA": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_RPPA.gz",
    "TCGA CNA": XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    "CPTAC-ind RNAseq": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct",
    "CPTAC-ind Proteome": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct",
    "CPTAC-ind Meth": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct",
    "CPTAC-ind CNA": CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_gistic_thresholded_tumor_v3.0.cct",
    "CPTAC-dis RNAseq": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct",
    "CPTAC-dis Proteome": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct",
    "CPTAC-dis Meth": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct",
    "CPTAC-dis SCNV": CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_SCNA_log2_gene_level.cct",
    "CPTAC-OV RNAseq": CPTAC_OV_DIR / "HS_CPTAC_OV_rnaseq_fpkm_log2.cct",
    "CPTAC-OV Proteome": CPTAC_OV_DIR / "HS_CPTAC_OV_proteome_gene_tumor.cct",
    "CPTAC-OV CNA": CPTAC_OV_DIR / "HS_CPTAC_OV_cnv_gene.cct",
}

# Groupings compared on the symbol axis. Layers that are not gene-level (450K
# probes, N-glycosylation sites) are deliberately absent.
GROUPS: dict[str, list[str]] = {
    "Expression (RNAseq/mRNA)": ["TCGA mRNA(EB++)", "TCGA mRNA(Hugo)", "CPTAC-ind RNAseq",
                                 "CPTAC-dis RNAseq", "CPTAC-OV RNAseq"],
    "Protein": ["TCGA RPPA", "CPTAC-ind Proteome", "CPTAC-dis Proteome", "CPTAC-OV Proteome"],
    "Copy number (CNA)": ["TCGA CNA", "CPTAC-ind CNA", "CPTAC-dis SCNV", "CPTAC-OV CNA"],
    "Methylation": ["CPTAC-ind Meth", "CPTAC-dis Meth"],
}

# Identifier shapes, tested in this order; the labels are the keys of the census
# written to work("alignment_ids").
SYMBOL_LIKE = re.compile(r"[A-Za-z][A-Za-z0-9\-\._@]*")
MIRNA_NAME = re.compile(r"hsa-[a-z0-9\-]+", re.I)


def open_text(path: Path):
    """Open a matrix for text reading, transparently handling gzip.

    Args:
        path: File to open.

    Returns:
        A text-mode file handle with replacement error handling.
    """
    return gzip.open(path, "rt", errors="replace") if str(path).endswith(".gz") else open(path, "rt", errors="replace")


def all_ids(path: Path) -> list[str]:
    """Return every row identifier of a matrix, in file order.

    Args:
        path: Tab-separated matrix; the first column holds the identifier and the
            first line is the header.

    Returns:
        The stripped, unquoted identifier of every data row.
    """
    ids: list[str] = []
    with open_text(path) as handle:
        first = True
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            if first:
                first = False
                continue
            ids.append(line.split("\t", 1)[0].strip().strip('"'))
    return ids


def census(ids: list[str]) -> tuple[collections.Counter, dict[str, str]]:
    """Count the identifier classes of a list of identifiers.

    Args:
        ids: Row identifiers of one matrix.

    Returns:
        A pair (counts per class, one example identifier per class).
    """
    counts: collections.Counter = collections.Counter()
    examples: dict[str, str] = {}
    for value in ids:
        if re.fullmatch(r"\d+", value):
            kind = "Entrez ID"
        elif value.upper().startswith("ENSG"):
            kind = "ENSG"
        elif re.fullmatch(r"cg\d+", value):
            kind = "450K probe"
        elif MIRNA_NAME.fullmatch(value):
            kind = "miRNA name"
        elif re.fullmatch(r"MIMAT\d+", value):
            kind = "miRNA(MIMAT)"
        elif SYMBOL_LIKE.fullmatch(value):
            kind = "symbol-like"
        else:
            kind = "other / locus ID"
        counts[kind] += 1
        examples.setdefault(kind, value)
    return counts, examples


def symbol_ids(tag: str) -> set[str]:
    """Return the symbol-like identifiers recorded for one matrix tag.

    Args:
        tag: Key of TARGETS.

    Returns:
        The subset of the matrix identifiers that look like gene symbols.
    """
    return {x for x in IDS.get(tag, set()) if SYMBOL_LIKE.fullmatch(x)}


# --- 1. Identifier census ------------------------------------------------------

CENSUS: dict[str, tuple] = {}
IDS: dict[str, set[str]] = {}
for tag, path in TARGETS.items():
    if not path.exists():
        CENSUS[tag] = ("file missing", {}, 0)
        continue
    identifiers = all_ids(path)
    CENSUS[tag] = (len(identifiers), dict(census(identifiers)[0]), census(identifiers)[1])
    IDS[tag] = set(identifiers)
    print("%-20s n=%-7d %s" % (tag, len(identifiers),
                               dict(sorted(census(identifiers)[0].items(), key=lambda x: -x[1]))))
    print("                     examples: %s" % census(identifiers)[1])

ids_path = work("alignment_ids")
ensure_dir(ids_path.parent)
with ids_path.open("w", encoding="utf-8") as handle:
    json.dump({k: {"n": v[0], "census": v[1], "examples": v[2]} for k, v in CENSUS.items()},
              handle, indent=1, ensure_ascii=False)

# --- 2. Illumina 450K probe -> gene mapping -----------------------------------

print("\n=== 450K probe -> gene mapping ===")
manifest = XENA_PANCAN_DIR / "illuminaMethyl450_hg19_GPL16304_TCGAlegacy"
probe2gene: dict[str, list[str]] = {}
with manifest.open(errors="replace") as handle:
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        genes = parts[1].strip()
        # The manifest uses "." for a probe that maps to no gene.
        if genes in ("", "."):
            continue
        probe2gene[parts[0].strip()] = [x for x in genes.split(",") if x]
probes = IDS.get("TCGA 450K", set())
mapped = [x for x in probes if x in probe2gene]
genes = {g for x in mapped for g in probe2gene[x]}
print("   manifest entries: %d" % len(probe2gene))
print("   TCGA-UCEC 450K probes: %d, of which mapped: %d (%.1f%%)"
      % (len(probes), len(mapped), 100 * len(mapped) / max(len(probes), 1)))
print("   unique genes reached: %d" % len(genes))
print("   example: cg00651829 -> %s" % probe2gene.get("cg00651829"))

probe_path = work("alignment_probe_map")
ensure_dir(probe_path.parent)
with probe_path.open("w", encoding="utf-8") as handle:
    json.dump({"probe2gene_entries": len(probe2gene), "ucec_probes": len(probes),
               "mapped": len(mapped), "genes": len(genes)}, handle, indent=1)

# --- 3. Common gene space on the symbol axis ----------------------------------

print("\n=== common gene space (symbol axis, non gene-level layers excluded) ===")
for group, tags in GROUPS.items():
    sets = {t: symbol_ids(t) for t in tags if t in IDS}
    sets = {k: v for k, v in sets.items() if v}
    if not sets:
        continue
    inter = set.intersection(*sets.values())
    print("\n  [%s]" % group)
    for tag, values in sets.items():
        print("     %-20s %6d" % (tag, len(values)))
    print("     >>> intersection of all = %d" % len(inter))

# The two intersections the benchmark actually needs: the three-cohort EC
# expression space and the two-cohort EC protein space.
for name, tags in [("three-cohort EC expression intersection", ["TCGA mRNA(Hugo)", "CPTAC-ind RNAseq", "CPTAC-dis RNAseq"]),
                   ("three-cohort EC protein intersection", ["CPTAC-ind Proteome", "CPTAC-dis Proteome"])]:
    sets = {t: symbol_ids(t) for t in tags if t in IDS}
    # Report only when every member of the group is present, otherwise the
    # intersection would be a misleading subset.
    if len(sets) == len(tags):
        print("\n  %s = %d" % (name, len(set.intersection(*sets.values()))))

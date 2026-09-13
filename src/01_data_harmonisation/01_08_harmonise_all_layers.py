#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the harmonised per-cohort, per-layer matrices.

Purpose
    Stage 01 (data harmonisation), final step and the one that produces the data
    every later stage reads. It loads each raw matrix, keeps only rows with a
    usable identifier, averages duplicated identifiers, restricts the TCGA
    pan-cancer features and samples to the UCEC whitelist, splits the CPTAC
    independent cohort into tumour and adjacent-normal columns using its metadata
    table, maps the 450K methylation probes to genes from the cached gene-level
    matrix, and writes one pickle per cohort x layer together with an inventory of
    shapes. Nothing is recomputed here: the identifier systems, the domains and the
    duplicates were characterised by probes 01_01 to 01_07, and this step applies
    those findings.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("xena_ucec_dir"), data("xena_pancan_dir") -- TCGA UCEC and pan-cancer
    matrices; the UCEC-specific GISTIC2 CNA matrix supplies the sample whitelist.
    data("cptac_independent_dir") -- CPTAC UCEC independent matrices plus
    UCEC_confirmatory_meta_table_v3.0.xlsx, whose Group and Case_id columns label
    each sample as tumour or normal.
    data("cptac_discovery_dir"), data("cptac_ov_dir") -- CPTAC UCEC discovery and
    OV matrices, tumour and normal files.
    work("harmonised_tcga_meth_genelevel") -- cached gene-level TCGA methylation
    matrix (probe-to-gene mapped). Must exist; the script reports it if not.

Outputs
    work("harmonised_pickle_pattern") -- one pickle per cohort x layer
    (for example TCGA__mRNA.pkl, ind__protein_N.pkl).
    work("harmonised_tmp_pattern") -- temporary cache of the pan-cancer column
    subset, deleted at the end of the run.
    work("harmonised_inventory") -- JSON, "cohort|layer" -> {shape, note}.

Usage
    python 01_08_harmonise_all_layers.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

# Make common.config importable no matter which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, data, ensure_dir, work

# Seeded once, as for every stochastic step of the package.
np.random.seed(SEED)

# --- Output locations read from config/paths.yaml ------------------------------

HARMONISED_DIR = ensure_dir(work("harmonised_dir"))
HARMONISED_PATTERN = str(work("harmonised_pickle_pattern"))
TMP_PATTERN = str(work("harmonised_tmp_pattern"))
METH_CACHE = work("harmonised_tcga_meth_genelevel")
INVENTORY_PATH = work("harmonised_inventory")

# --- Input locations read from config/paths.yaml ------------------------------

XENA_PANCAN_DIR = data("xena_pancan_dir")
XENA_UCEC_DIR = data("xena_ucec_dir")
CPTAC_DISCOVERY_DIR = data("cptac_discovery_dir")
CPTAC_INDEPENDENT_DIR = data("cptac_independent_dir")
CPTAC_OV_DIR = data("cptac_ov_dir")

# A usable identifier line: alphanumeric start, then the characters used by gene
# symbols, Ensembl IDs and LinkedOmics site identifiers.
USABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\._@/]*$")

# Inventories written to work("harmonised_inventory"), keyed by "cohort|layer".
INVENTORY: dict[str, dict] = {}


def load(path: Path, cols: list[int] | None = None) -> pd.DataFrame:
    """Load a tab-separated matrix into a numeric frame indexed by identifier.

    Args:
        path: Matrix file, plain or gzip-compressed.
        cols: Column positions to read, positional as accepted by pandas; None
            reads every column. Used to subset the wide pan-cancer matrices
            before they are parsed.

    Returns:
        Feature x sample DataFrame of floats, indexed by identifier, with
        duplicated identifiers averaged.
    """
    df = pd.read_csv(path, sep="\t", comment="#", header=0, dtype=str, engine="python", usecols=cols)
    df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip().str.strip('"')
    # Rows whose identifier column is empty or malformed cannot be joined to
    # anything, so they are removed before the frame becomes numeric.
    df = df[df["ID"].str.match(USABLE_ID, na=False)]
    matrix = df.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return matrix.groupby(level=0).mean()


def normalise_sample_id(value) -> str:
    """Reduce a sample column label to its TCGA barcode core.

    Args:
        value: Any column label, for example "TCGA-BK-A0CC-01A" or
            "TCGA.BK.A0CC.01".

    Returns:
        The "TCGA-XX-YYYY" barcode prefix when the label carries one, otherwise the
        upper-cased, stripped label unchanged.
    """
    match = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(value).upper().strip())
    return match.group(1) if match else str(value).strip()


def load_tcga_pancan(path: Path, whitelist: set[str]) -> pd.DataFrame:
    """Subset a pan-cancer Xena matrix to the UCEC sample whitelist.

    Args:
        path: Gzip-compressed pan-cancer matrix.
        whitelist: Normalised barcodes of the samples to keep.

    Returns:
        Feature x sample DataFrame restricted to the whitelist, with columns
        deduplicated by averaging.

    Notes:
        The pan-cancer matrix carries tens of thousands of sample columns, so only
        its header is read up front and the retained column positions are handed to
        pandas. The result is cached on disk, because re-reading the full matrix
        for every re-run is the most expensive step of the stage.
    """
    cache_path = Path(TMP_PATTERN.format(name=Path(path).name))
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    with gzip.open(path, "rt", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    keep = [i for i, column in enumerate(header) if i > 0 and normalise_sample_id(column) in whitelist]
    matrix = load(path, cols=[0] + keep)
    matrix.columns = [normalise_sample_id(c) for c in matrix.columns]
    # Several columns can normalise to the same barcode (different aliquots of the
    # same sample); averaging keeps the sample axis unique.
    matrix = matrix.T.groupby(level=0).mean().T
    matrix.to_pickle(cache_path)
    return matrix


def load_independent_layer(path: Path, split: bool = False):
    """Load a CPTAC-UCEC independent layer, optionally split into T/N.

    Args:
        path: LinkedOmics matrix for the independent cohort.
        split: When True, split the samples by the Group column of the cohort
            metadata table into tumour and normal frames.

    Returns:
        The whole frame when ``split`` is False, otherwise the pair
        (tumour frame, normal frame).
    """
    matrix = load(path)
    if split:
        # Sample labels carry the cohort suffix (for example "-A" for adjacent
        # normal), which is stripped before the metadata lookup.
        groups = {c: I_GRP.get(re.sub(r"-[A-Z]$", "", str(c).upper())) for c in matrix.columns}
        tumour = [c for c, g in groups.items() if g == "Tumor"]
        normal = [c for c, g in groups.items() if g in ("Adjacent_normal", "Enriched_Normal")]
        return (matrix[tumour].dropna(axis=1, how="all"),
                matrix[normal].dropna(axis=1, how="all"))
    return matrix


def put(cohort: str, layer: str, matrix: pd.DataFrame, note: str = "") -> None:
    """Write one harmonised cohort x layer matrix and record it in the inventory.

    Args:
        cohort: Short cohort key ("TCGA", "ind", "dis", "OV").
        layer: Layer key ("mRNA", "protein", "miRNA", "CNA", "meth").
        matrix: Feature x sample DataFrame.
        note: Optional note stored with the inventory record and printed.

    Returns:
        None. The matrix is written to work("harmonised_pickle_pattern").
    """
    matrix = matrix.dropna(axis=1, how="all")
    record: dict = {"shape": list(matrix.shape)}
    if note:
        record["note"] = note
    INVENTORY[f"{cohort}|{layer}"] = record
    print(f"  {cohort:5s}/{layer:9s} {matrix.shape[0]:6d} genes x {matrix.shape[1]:4d} samples  {note}")
    matrix.to_pickle(HARMONISED_PATTERN.format(cohort=cohort, layer=layer))


# --- UCEC sample whitelist -----------------------------------------------------

print("=" * 92)
print("[build]")
print("=" * 92)

# The UCEC-specific CNA matrix is the only TCGA-UCEC asset that carries exactly the
# cohort's own sample set, so its column names define the whitelist applied to the
# pan-cancer matrices.
ucec_cna = load(XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz")
UWH = {normalise_sample_id(c) for c in ucec_cna.columns}
print(f"UCEC whitelist: {len(UWH)} samples (from the UCEC-specific CNA column names)")

# --- TCGA-UCEC -----------------------------------------------------------------

matrix = load_tcga_pancan(XENA_PANCAN_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz", UWH)
put("TCGA", "mRNA", matrix, "(pan-cancer EB++, columns subset to the UCEC whitelist)")

matrix = load(XENA_UCEC_DIR / "TCGA.UCEC.sampleMap_miRNA_HiSeq_gene.gz")
matrix.columns = [normalise_sample_id(c) for c in matrix.columns]
put("TCGA", "miRNA", matrix)

matrix = ucec_cna.copy()
matrix.columns = [normalise_sample_id(c) for c in matrix.columns]
put("TCGA", "CNA", matrix)

# --- CPTAC-UCEC independent ----------------------------------------------------

# Group of every independent-cohort case, read from the cohort metadata workbook.
workbook = openpyxl.load_workbook(CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_meta_table_v3.0.xlsx",
                                  read_only=True)
sheet = workbook[workbook.sheetnames[0]]
meta_rows = list(sheet.iter_rows(values_only=True))
meta_header = [str(x) for x in meta_rows[0]]
group_idx = meta_header.index("Group")
case_idx = meta_header.index("Case_id")
I_GRP = {str(r[case_idx]).strip().upper(): str(r[group_idx]) for r in meta_rows[1:] if r[case_idx]}

independent_rna = CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_RNAseq_gene_RSEM_removed_circRNA_UQ_log2(x+1)_tumor_normal_v3.0.cct"
independent_protein = CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_proteomics_ratio_median_polishing_log2_tumor_normal_v3.0.cct"
independent_mirna = CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_miRNAseq_miRNA_TPM_log2(x+1)_tumor_normal_v3.0.cct"

tumour, normal = load_independent_layer(independent_rna, True)
put("ind", "mRNA", tumour)
put("ind", "mRNA_N", normal)
tumour, normal = load_independent_layer(independent_protein, True)
put("ind", "protein", tumour)
put("ind", "protein_N", normal)
tumour, _normal = load_independent_layer(independent_mirna, True)
put("ind", "miRNA", tumour)
put("ind", "CNA", load(CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_WGS_cnv_log2_ratio_tumor_v3.0.cct"))
put("ind", "meth", load(CPTAC_INDEPENDENT_DIR / "UCEC_confirmatory_methylation_gene_level_beta_value_tumor_v3.0.cct"))

# --- CPTAC-UCEC discovery ------------------------------------------------------

put("dis", "mRNA", load(CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Tumor.cct"))
put("dis", "mRNA_N", load(CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_RNAseq_RSEM_UQ_log2_Normal.cct"))
put("dis", "protein", load(CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Tumor.cct"))
put("dis", "protein_N", load(CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Proteomics_TMT_gene_level_Normal.cct"))
put("dis", "miRNA", load(CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_microRNA_log2_Tumor.cct"))
put("dis", "CNA", load(CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_SCNA_log2_gene_level.cct"))
put("dis", "meth", load(CPTAC_DISCOVERY_DIR / "HS_CPTAC_UCEC_Methylation_betaValue_gene_level_mean.cct"))

# --- CPTAC-OV prospective ------------------------------------------------------

put("OV", "mRNA", load(CPTAC_OV_DIR / "HS_CPTAC_OV_rnaseq_fpkm_log2.cct"))
put("OV", "CNA", load(CPTAC_OV_DIR / "HS_CPTAC_OV_cnv_gene.cct"))
put("OV", "protein", load(CPTAC_OV_DIR / "HS_CPTAC_OV_proteome_gene_tumor.cct"))
put("OV", "protein_N", load(CPTAC_OV_DIR / "HS_CPTAC_OV_proteome_gene_normal.cct"))

# --- TCGA methylation, probe-to-gene mapped ------------------------------------

if METH_CACHE.exists():
    matrix = pd.read_csv(METH_CACHE, index_col=0)
    matrix.columns = [normalise_sample_id(x) for x in matrix.columns]
    matrix = matrix.T.groupby(level=0).mean().T
    put("TCGA", "meth", matrix, "(450K probes mapped to genes, read from the local cache)")
else:
    print(f"  !! TCGA/meth cache missing at {METH_CACHE}; run the probe-to-gene step first")

ensure_dir(INVENTORY_PATH.parent)
with INVENTORY_PATH.open("w", encoding="utf-8") as handle:
    json.dump(INVENTORY, handle, indent=1, ensure_ascii=False)
print(f"\nCached {len(INVENTORY)} layers")

# The pan-cancer column subsets are throw-away caches that would otherwise make
# the work directory grow by gigabytes.
for entry in HARMONISED_DIR.iterdir():
    if entry.name.startswith("_tmp_"):
        entry.unlink()
print("Temporary files removed")

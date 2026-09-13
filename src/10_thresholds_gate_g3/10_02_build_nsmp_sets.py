#!/usr/bin/env python3
"""
Build the NSMP sample sets of the three cohorts and check their layer coverage.

Purpose
    Gate G3 measures how reproducible the copy-number-low (NSMP) subtype is, and
    that measurement is only meaningful for the samples that carry every layer of
    the analysis. This module resolves the NSMP participant list of each of the
    three cohorts, intersects it with the sample identifiers of the four
    harmonised layer matrices, and writes the resulting sample sets together with
    their counts. It is the second step of the stage: the sets it writes are what
    the identifier alignment, the reproducibility measurement and the document
    package afterwards operate on.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("core_nsmp_samples") -- JSON catalogue of the frozen document package;
        holds a core participant list and the NSMP subset of it, whose key names
        are params:nsmp.catalogue_core_key and params:nsmp.catalogue_nsmp_key.
    data("cptac_independent_metadata") and data("cptac_discovery_clinical") --
        metadata tables carrying the molecular-subtype annotation of the two
        CPTAC cohorts.
    work("harmonised_matrix_pattern") -- the harmonised per-cohort per-layer
        matrices, one file per cohort tag and layer suffix.

Outputs
    work("nsmp_sets") -- JSON with the NSMP participants of each cohort, the
        four-layer intersection of each, and the four counts.

Usage
    python src/10_thresholds_gate_g3/10_02_build_nsmp_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Allow the module to be run from any directory: put src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, param, work  # noqa: E402

# Number of participants below which a list stored in the catalogue is too short
# to be the NSMP participant list; the catalogue also holds derived sub-lists and
# counters, so the list is recognised by being long and made of identifier
# strings.
MIN_CATALOGUE_LIST_LENGTH = 30


def layer_file_suffixes() -> dict[str, str]:
    """Return layer name -> harmonised file-name suffix for the layers used here.

    The four-layer intersection uses the layers the study integrates across
    cohorts; their file-name suffixes are the values of params:layer_tags, which
    is why the suffix vocabulary is not repeated in code.

    Returns:
        A dict such as {"mRNA": "mRNA", "methylation": "meth"}, in the order of
        params:integrable_layers.
    """
    layer_tags = param("layer_tags")
    return {name: layer_tags[name] for name in param("integrable_layers")}


def layer_matrix_path(cohort_tag: str, layer_suffix: str) -> Path:
    """Return the harmonised matrix path of one cohort and one layer.

    Args:
        cohort_tag: tag of the cohort, a value of params:cohort_tags.
        layer_suffix: file-name suffix of the layer, a value of params:layer_tags.

    Returns:
        The resolved path of the matrix file.
    """
    pattern = str(work("harmonised_matrix_pattern"))
    return Path(pattern.format(cohort=cohort_tag, layer=layer_suffix))


def load_layer_matrix(cohort_tag: str, layer_suffix: str) -> pd.DataFrame | None:
    """Load one harmonised layer matrix, or None when the file does not exist.

    Args:
        cohort_tag: tag of the cohort.
        layer_suffix: file-name suffix of the layer.

    Returns:
        The matrix with samples in the columns, or None when the cohort has no
        matrix for that layer. Column names are cast to ``str`` because the
        harmonised identifiers are compared as strings.
    """
    path = layer_matrix_path(cohort_tag, layer_suffix)
    if not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix


def read_metadata_table(path: Path) -> pd.DataFrame:
    """Read a cohort metadata table.

    Args:
        path: path of the table, as declared in paths.yaml.

    Returns:
        The table. The reader follows the file suffix: the confirmatory cohort
        publishes an Excel workbook, the discovery cohort a tab-separated text
        export.
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")


def read_nsmp_catalogue() -> dict:
    """Load the frozen NSMP catalogue of the document package.

    Returns:
        The parsed JSON catalogue, whose keys are declared in
        paths.yaml -> work.core_nsmp_samples.
    """
    with work("core_nsmp_samples").open(encoding="utf-8") as handle:
        return json.load(handle)


def report_catalogue(catalogue) -> list | None:
    """Print the structure of the catalogue and return its NSMP participant list.

    Args:
        catalogue: the parsed NSMP catalogue.

    Returns:
        The participant list, or None when the catalogue holds no list that looks
        like one. The list is recognised -- not read from a declared key -- so
        that the structure of the catalogue is what decides, and the entry used is
        printed.
    """
    print("type:", type(catalogue).__name__, "| keys/length:",
          list(catalogue.keys())[:6] if isinstance(catalogue, dict) else len(catalogue))
    identifiers = None
    if isinstance(catalogue, dict):
        for key, value in catalogue.items():
            length = len(value) if hasattr(value, "__len__") else value
            print(f"  {key}: {type(value).__name__} length {length}")
            if isinstance(value, list) and value:
                print(f"     sample {value[:4]}")
        for key, value in catalogue.items():
            if (isinstance(value, list) and len(value) > MIN_CATALOGUE_LIST_LENGTH
                    and all(isinstance(entry, str) for entry in value[:5])):
                identifiers = value
                print(f"  -> used as the NSMP list: key '{key}', {len(value)} participants")
                break
    else:
        identifiers = catalogue
        print("  sample:", list(identifiers)[:4])
    return identifiers


def nsmp_identifiers_from_metadata(spec: dict) -> set[str]:
    """Return the NSMP participants of a cohort read from its metadata table.

    Args:
        spec: the cohort entry of params:nsmp.cohorts, giving the metadata table,
            the participant column, the subtype column and the NSMP class.

    Returns:
        The set of participant identifiers whose subtype column carries the NSMP
        class of that cohort.
    """
    table = read_metadata_table(data(spec["metadata_file"]))
    selected = table.loc[table[spec["subtype_column"]] == spec["nsmp_value"], spec["participant_column"]]
    return set(selected.astype(str))


def intersect_layers_reporting_column_counts(
    cohort_tag: str, identifiers: set[str], layer_suffixes: dict[str, str]
) -> set[str]:
    """Intersect one cohort's NSMP list with the columns of every layer.

    Used for the cohort whose list comes from the frozen catalogue, where the size
    of each layer is reported next to the number of NSMP participants it covers,
    so that a small set is not mistaken for a missing layer.

    Args:
        cohort_tag: tag of the cohort.
        identifiers: the candidate NSMP participants.
        layer_suffixes: layer name -> file-name suffix.

    Returns:
        The identifiers present in every available layer. A layer whose matrix file
        is absent is reported and skipped.
    """
    intersection = identifiers.copy()
    for suffix in layer_suffixes.values():
        matrix = load_layer_matrix(cohort_tag, suffix)
        if matrix is None:
            print(f"  {suffix}: layer missing")
            continue
        hits = identifiers & set(matrix.columns)
        print(f"  {suffix:7s}: layer columns {matrix.shape[1]:4d} | NSMP hits {len(hits):3d}")
        intersection &= set(matrix.columns)
    return intersection


def intersect_layers(cohort_tag: str, identifiers: set[str], layer_suffixes: dict[str, str]) -> set[str]:
    """Intersect one cohort's NSMP list with the columns of every layer.

    Used for the two smaller cohorts, where only the number of NSMP participants
    per layer is reported.

    Args:
        cohort_tag: tag of the cohort.
        identifiers: the candidate NSMP participants.
        layer_suffixes: layer name -> file-name suffix.

    Returns:
        The identifiers present in every available layer. A layer whose matrix file
        is absent is reported and skipped.
    """
    intersection = identifiers.copy()
    for suffix in layer_suffixes.values():
        matrix = load_layer_matrix(cohort_tag, suffix)
        if matrix is None:
            print(f"  {suffix}: layer missing")
            continue
        hits = identifiers & set(matrix.columns)
        print(f"  {suffix:7s}: NSMP hits {len(hits):3d}")
        intersection &= set(matrix.columns)
    return intersection


def main() -> None:
    """Build the NSMP sets of the three cohorts and write them out.

    Returns:
        None. The sets are written to work("nsmp_sets").
    """
    layer_suffixes = layer_file_suffixes()
    cohort_specs = param("nsmp", "cohorts")
    cohort_tags = param("cohort_tags")

    print("=== Previous TCGA NSMP artefact ===")
    catalogue = read_nsmp_catalogue()
    identifiers = report_catalogue(catalogue)

    sets: dict[str, list[str]] = {}
    counts: dict[str, int] = {}

    print("\n=== TCGA NSMP availability in each layer ===")
    catalogue_cohort = next(label for label, spec in cohort_specs.items()
                            if spec["nsmp_source"] == "catalogue")
    tcga_spec = cohort_specs[catalogue_cohort]
    tcga_tag = cohort_tags[tcga_spec["cohort_tag"]]
    tcga = set(identifiers) if identifiers else set()
    print(f"  NSMP list: {len(tcga)} participants")
    # The full NSMP list is stored as well as its four-layer intersection: the
    # difference between the two is the coverage the cohort loses, which is part
    # of what the threshold derivation has to account for.
    tcga_intersection = intersect_layers_reporting_column_counts(tcga_tag, tcga, layer_suffixes)
    print(f"  ** four-layer NSMP intersection: {len(tcga_intersection)} participants")
    sets[f"{catalogue_cohort}_nsmp_all"] = sorted(tcga)
    sets[f"{catalogue_cohort}_nsmp_4layer"] = sorted(tcga_intersection)
    counts[f"{catalogue_cohort}_list"] = len(tcga)
    counts[f"{catalogue_cohort}_4L"] = len(tcga_intersection)

    for label, spec in cohort_specs.items():
        if spec["nsmp_source"] == "catalogue":
            continue
        tag = cohort_tags[spec["cohort_tag"]]
        print(f"\n=== {label} (CPTAC {tag}) NSMP ===")
        cohort_ids = nsmp_identifiers_from_metadata(spec)
        print(f"  {spec['subtype_column']} == {spec['nsmp_value']}: {len(cohort_ids)} participants")
        intersection = intersect_layers(tag, cohort_ids, layer_suffixes)
        print(f"  ** four-layer intersection: {len(intersection)} participants")
        sets[f"{label}_nsmp_4layer"] = sorted(intersection)
        counts[f"{label}_4L"] = len(intersection)

    out = {
        **sets,
        "counts": counts,
    }
    with work("nsmp_sets").open("w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print(f"\nWrote {work('nsmp_sets')}")
    print("counts:", out["counts"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Resolve the TCGA sample identifiers against the matrix columns of the NSMP work.

Purpose
    Gate G3 compares the copy-number-low (NSMP) subtype across three cohorts, and
    the TCGA arm of that comparison is identified by a participant list while its
    harmonised layer matrices are keyed by their own column names. This module is
    the identifier alignment step of the stage: it loads the frozen NSMP catalogue,
    reports how the NSMP subset relates to the core sample set, and then, for every
    integrable layer of TCGA-UCEC, reads the column names of the harmonised matrix
    and counts how many NSMP and how many core identifiers they match, both on the
    full identifier and on its leading barcode prefix. Those counts establish
    whether the two identifier systems can be joined at all, which the NSMP sample
    sets of the following step depend on. Nothing is written; the report is the
    answer.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("core_nsmp_samples")
        JSON catalogue of the frozen document package: the core participant list
        under params:nsmp.catalogue_core_key and the NSMP subset under
        params:nsmp.catalogue_nsmp_key.
    work("harmonised_matrix_pattern")
        Harmonised per-cohort per-layer matrices. The TCGA column names are read
        for every layer of params:integrable_layers, through params:cohort_tags
        for the cohort and params:layer_tags for the file-name suffix.

Outputs
    None. The identifier alignment is printed; no artefact is written.

Usage
    python src/10_thresholds_gate_g3/10_03_resolve_tcga_identifiers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Allow the module to be run from any directory: put src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import param, work  # noqa: E402


def layer_file_suffixes() -> dict[str, str]:
    """Return layer name -> harmonised file-name suffix for the integrable layers.

    The suffix vocabulary is the values of params:layer_tags, which is why the file
    names of the matrices are not repeated in code.

    Returns:
        A dict such as {"mRNA": "mRNA", "methylation": "meth"}, in the order of
        params:integrable_layers.
    """
    layer_tags = param("layer_tags")
    return {name: layer_tags[name] for name in param("integrable_layers")}


def tcga_matrix_path(layer_suffix: str) -> Path:
    """Return the harmonised TCGA-UCEC matrix path of one layer.

    Args:
        layer_suffix: file-name suffix of the layer, a value of params:layer_tags.

    Returns:
        The resolved path of the matrix file.
    """
    pattern = str(work("harmonised_matrix_pattern"))
    cohort_tag = param("cohort_tags")["tcga"]
    return Path(pattern.format(cohort=cohort_tag, layer=layer_suffix))


def read_nsmp_catalogue() -> tuple[set, set]:
    """Load the frozen NSMP catalogue and split it into its two participant sets.

    Returns:
        A pair ``(core, nsmp)``: the full core set of the catalogue and the NSMP
        (copy-number low) subset of it. The key names are read from
        params:nsmp.catalogue_core_key and params:nsmp.catalogue_nsmp_key, as 10_02
        does, so both steps read the same two lists.
    """
    with work("core_nsmp_samples").open(encoding="utf-8") as handle:
        catalogue = json.load(handle)
    core = set(catalogue[param("nsmp", "catalogue_core_key")])
    nsmp = set(catalogue[param("nsmp", "catalogue_nsmp_key")])
    return core, nsmp


def report_matches(columns: list, identifiers: set, label: str, prefix_length: int) -> None:
    """Print how many of the matrix columns match one identifier list.

    Args:
        columns: the column names of one harmonised matrix, as strings.
        identifiers: the participant identifiers to match against.
        label: name of the identifier list, used in the report line.
        prefix_length: number of leading characters the two identifier systems are
            compared on.

    Returns:
        None. The hit count is written to standard output.
    """
    # Matching on the barcode prefix is what allows the comparison at all: the
    # harmonised matrices carry the full barcode while the catalogue carries a
    # shorter identifier for some cohorts, so the prefix is the common part.
    prefix_set = {identifier[:prefix_length] for identifier in identifiers}
    hits = len({column[:prefix_length] for column in columns} & prefix_set)
    print(f"  matched on the first {prefix_length} characters, {label}: hits {hits}")


def main() -> None:
    """Report the identifier alignment of every integrable TCGA-UCEC layer.

    Returns:
        None. Everything is written to standard output.
    """
    core, nsmp = read_nsmp_catalogue()
    # The NSMP list is expected to be a subset of the core list; when it is not,
    # the catalogue mixes identifier systems and the counts below cannot be trusted.
    print(f"core {len(core)} | nsmp_core {len(nsmp)} | nsmp subset of core ? {nsmp <= core}")
    # Four identifiers are enough to recognise the naming scheme in the log.
    print("NSMP sample:", sorted(nsmp)[:4])

    prefix_length = param("nsmp", "barcode_prefix_length")
    for layer_name, suffix in layer_file_suffixes().items():
        path = tcga_matrix_path(suffix)
        if not path.exists():
            print(f"  {layer_name}: missing")
            continue
        # The header alone is read: only the column names are needed and the
        # harmonised matrices are wide.
        matrix = pd.read_csv(path, index_col=0, nrows=0)
        columns = [str(column) for column in matrix.columns]
        print(f"\n{layer_name}: columns {len(columns)}")
        # First and last columns identify the barcode convention (which cohort part
        # the identifier names) without printing hundreds of names.
        print("  first 3:", columns[:3])
        print("  last 3:", columns[-3:])
        report_matches(columns, nsmp, f"nsmp ({prefix_length}-char)", prefix_length)
        report_matches(columns, core, f"core ({prefix_length}-char)", prefix_length)


if __name__ == "__main__":
    main()

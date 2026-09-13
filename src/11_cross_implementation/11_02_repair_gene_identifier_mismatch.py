#!/usr/bin/env python3
"""
Cross-implementation entry point for the EEEC gene-identifier repair step.

Purpose
    Stage 08 repairs the EEEC protein-to-gene map with
    08_02_repair_gene_identifier_mismatch.py. Stage 11 compares the Python and
    the R toolchains, and its first step consumes the same repaired map, so the
    historical layout had the identical repair script in both stage
    directories. The two legacy copies were byte-identical, so the stage-11
    location is kept only as an entry point: it loads the stage-08
    implementation by file path and runs it. There is one implementation, and
    the two entry points exist so that a reader of either stage can start the
    repair from the directory they are working in. Running this file has exactly
    the same effect as running the stage-08 script.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    None directly. The loaded implementation reads the same configuration
    entries as the stage-08 script.

Outputs
    None directly. work("eeec_protein_gene_map") is rewritten in place by the
    loaded implementation.

Usage
    python src/11_cross_implementation/11_02_repair_gene_identifier_mismatch.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# The single implementation of the repair lives in stage 08; this stage only
# forwards to it, so there is no second copy to keep in sync.
IMPLEMENTATION = (Path(__file__).resolve().parents[1]
                  / "08_cross_platform"
                  / "08_02_repair_gene_identifier_mismatch.py")


def load_implementation():
    """Load the stage-08 repair module from its file path.

    The file name starts with digits, so the module cannot be imported by name;
    importlib is used to load it from the resolved path instead.

    Args:
        None.

    Returns:
        The loaded module object, which exposes the function main().

    Raises:
        FileNotFoundError: when the stage-08 implementation is missing.
    """
    if not IMPLEMENTATION.exists():
        raise FileNotFoundError(f"Repair implementation not found: {IMPLEMENTATION}")
    spec = importlib.util.spec_from_file_location("repair_gene_identifier_mismatch", IMPLEMENTATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Run the stage-08 gene-identifier repair through the stage-11 entry point.

    Args:
        None.

    Returns:
        None. The repair is executed with the arguments the stage-08
        implementation expects.
    """
    load_implementation().main()


if __name__ == "__main__":
    main()

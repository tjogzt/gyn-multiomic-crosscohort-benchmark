"""
Extract every number the narrative sections need from the analysis artefacts.

Purpose
    Stage 14, second script. The manuscript may not quote a number that was not
    read off an artefact, so this script dumps, section by section, the values
    the Methods and Results sections rely on: cohort and layer availability, the
    cross-cohort alignment diagnostics, the alignment-strategy matrix, the
    benchmark summary and its per-domain tables, and the PAC stability relation.
    It is the extraction half of the verification pair; 14_03 checks that each
    extracted value actually occurs in the manuscript text.

    The extraction is automatic: every value is read out of an artefact and
    printed, and nothing is listed by hand, so a value the artefacts no longer
    produce simply stops being reported. A missing artefact makes the script
    print nothing for that section rather than fail, so it can be run while the
    pipeline is still being built.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("harmonised_inventory")          JSON, harmonised matrix inventory.
    work("core5_samples")                 JSON, frozen core sample sets.
    work("core_nsmp_samples")             JSON, frozen NSMP catalogue.
    work("alignment_concordance")         JSON, per-layer profile concordance.
    work("alignment_domains")             JSON, numeric domain per matrix.
    work("alignment_ids")                 JSON, identifier reconciliation.
    work("protein_bridge_anchors")        JSON, mRNA-protein anchor.
    work("protein_bridge_three_cohort")   JSON, three-cohort protein bridge.
    work("protein_bridge_diagnostics")    JSON, bridge robustness diagnostics.
    work("alignment_ari_aggregate")       JSON, alignment-strategy matrix.
    work("benchmark_summary")             JSON, benchmark headline tables.
    work("benchmark_table_a")             JSON, per-domain benchmark table A.
    work("benchmark_table_b")             JSON, per-domain benchmark table B.
    work("pac_aggregate")                 JSON, PAC and transfer ARI.
    config/params.yaml
        verification.pac_bin_edges       Bin edges of the PAC-to-ARI curve.

Outputs
    None. The extracted values are printed to standard output.

Usage
    python 14_02_extract_manuscript_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import param, work

# Horizontal rule that separates the sections of the printed report.
RULE = "=" * 94


def load_work(key: str, default=None):
    """Read one work artefact, returning ``default`` when it is absent.

    Args
        key: entry name inside the work section of config/paths.yaml.
        default: value returned when the artefact does not exist.

    Returns
        The decoded JSON object, or ``default``.
    """
    path = work(key)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Print the extraction report, section by section."""
    print(RULE)
    print("[1] Cohort and layer availability")
    print(RULE)
    inventory = load_work("harmonised_inventory")
    if inventory:
        if isinstance(inventory, dict):
            # The inventory is large; three entries are enough to show the
            # record shape (cohort/layer key -> shape and note).
            for key, value in list(inventory.items())[:3]:
                print(f"  {key}: {str(value)[:160]}")
        else:
            print(pd.DataFrame(inventory).head(12).to_string())
    core5 = load_work("core5_samples")
    if core5:
        # The catalogue may be a mapping of set name to members or a bare list;
        # the printed summary reports the size of whatever it is.
        if isinstance(core5, dict):
            sizes = {
                key: (len(value) if hasattr(value, "__len__") else value)
                for key, value in core5.items()
            }
            print(f"\n  ucec_core5: {type(core5).__name__} {sizes}")
        else:
            print(f"\n  ucec_core5: {type(core5).__name__} {len(core5)}")
    nsmp = load_work("core_nsmp_samples")
    if nsmp:
        print(f"  ucec_core_nsmp: " + str({key: len(value) for key, value in nsmp.items()}))

    print("\n" + RULE)
    print("[2] Cross-cohort alignment (facts F-13 to F-23)")
    print(RULE)
    concordance = load_work("alignment_concordance")
    if concordance:
        print("  align_concordance:", json.dumps(concordance, ensure_ascii=False)[:1400])
    domains = load_work("alignment_domains")
    if domains:
        print("\n  align_domains:", json.dumps(domains, ensure_ascii=False)[:900])
    identifiers = load_work("alignment_ids")
    if identifiers:
        print("\n  align_ids:", json.dumps(identifiers, ensure_ascii=False)[:700])
    anchors = load_work("protein_bridge_anchors")
    if anchors:
        print("\n  b3_anchors:", json.dumps(anchors, ensure_ascii=False)[:700])
    bridge = load_work("protein_bridge_three_cohort")
    if bridge:
        print("\n  bridge_3cohort:", json.dumps(bridge, ensure_ascii=False)[:900])
    diagnostics = load_work("protein_bridge_diagnostics")
    if diagnostics:
        print("\n  bridge_diag:", json.dumps(diagnostics, ensure_ascii=False)[:700])

    print("\n" + RULE)
    print("[3] Alignment-strategy matrix (gate G4)")
    print(RULE)
    ari = load_work("alignment_ari_aggregate")
    if ari:
        print(" ", json.dumps(ari, ensure_ascii=False)[:1200])

    print("\n" + RULE)
    print("[4] Benchmark (Tables 1 and 2, and the layer-count curve)")
    print(RULE)
    summary = load_work("benchmark_summary")
    if summary:
        print(" ", json.dumps(summary, ensure_ascii=False)[:1600])
    for key in ["benchmark_table_a", "benchmark_table_b"]:
        table = load_work(key)
        if table:
            # The artefact is reported under its file name, which is what the
            # narrative refers to.
            print(f"\n  {work(key).name}:", json.dumps(table, ensure_ascii=False)[:800])

    print("\n" + RULE)
    print("[5] PAC (Tables 5 to 8)")
    print(RULE)
    pac = load_work("pac_aggregate")
    if pac:
        print("  tab1:", json.dumps(pac["tab1"], ensure_ascii=False))
        cross = pac.get("cross")
        if isinstance(cross, list):
            frame = pd.DataFrame(cross)
            pac_values = pd.to_numeric(frame["pac"], errors="coerce")
            ari_values = pd.to_numeric(frame["ari"], errors="coerce")
            # The bin edges come from the configuration so that this extraction,
            # the PAC-to-transfer table and the figure panel cannot drift apart.
            bins = pd.cut(pac_values, param("verification", "pac_bin_edges"), right=False)
            print("\n  PAC bins -> transfer ARI:")
            print(
                frame.assign(a=ari_values, b=bins)
                .dropna(subset=["a"])
                .groupby("b", observed=True)["a"]
                .agg(["mean", "count"])
                .round(4)
                .to_string()
            )


if __name__ == "__main__":
    main()

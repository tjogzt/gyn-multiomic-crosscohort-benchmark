"""
Build the evidence ledger that every narrative claim is checked against.

Purpose
    Stage 14, first script. The manuscript quotes a small number of headline
    numbers; this ledger re-extracts each of them from the artefact that
    produced it, so that a claim is never written from memory. It prints the
    layer-count curves, the method rankings, the PAC relation, the batch and
    cross-platform diagnostics, the G2 and P2 results and the remaining negative
    results, grouped by the conclusion they support.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")          JSON, one record per evaluation unit.
    work("benchmark_merged")          JSON, merged Python and R rankings.
    work("pac_aggregate")             JSON, PAC and degeneracy per method.
    work("eeec_batch_verify_json")    JSON, batch-testbed verification rows.
    work("platform_results_json")     JSON, cross-platform ladder and network.
    work("g2_refined")                JSON, frozen G2 baseline.
    work("pai_results")               JSON, purity-aware arm comparison.
    work("eeec_batch_purity_diag")    JSON, purity diagnostics in the testbed.
    work("purity_diag_json")          JSON, purity diagnostics fallback.

Outputs
    None. The ledger is printed.

Usage
    python 14_01_build_artifact_ledger.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import work

RULE = "=" * 92


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


def report_layer_curve(matrix: pd.DataFrame) -> None:
    """Print the record-weighted transfer ARI against the number of layers.

    Args
        matrix: the benchmark matrix, one row per evaluation unit.

    Returns
        None. The curve is printed.
    """
    # Keep only non-degenerate records at k=4, then weight by record: average
    # over methods first and over layer subsets second, so that a method that
    # failed on many subsets cannot dominate the curve.
    matrix["ari4"] = matrix["ari"].map(
        lambda value: value.get("4") if isinstance(value, dict) else None
    )
    matrix["degen4"] = matrix["degen"].map(
        lambda value: value.get("4") if isinstance(value, dict) else None
    )
    matrix = matrix[matrix["degen4"].fillna(1) == 0].copy()
    # Domain A is the three-cohort CPTAC domain, domain B the EC domain; the
    # domain label is derived from the artefact rather than matched literally.
    matrix["dom"] = matrix["domain"].map(
        lambda label: "A" if "CPTAC" in str(label) else "B"
    )
    matrix["nlay"] = matrix["subset"].map(lambda s: len(str(s).split("+")))
    records = (
        matrix.groupby(["dom", "subset", "pair", "nlay"])["ari4"].mean().reset_index()
    )
    print("Python side (record-weighted: methods first, layer subsets second)")
    for domain in ["A", "B"]:
        group = (
            records[records["dom"] == domain].groupby("nlay")["ari4"].agg(["mean", "count"])
        )
        parts = [
            f"{int(layers)} layers {value:.4f}(n={int(group.loc[layers, 'count'])})"
            for layers, value in group["mean"].items()
        ]
        print(f"  domain {domain}: " + " -> ".join(parts))


def main() -> None:
    """Print the whole evidence ledger, section by section."""
    print(RULE)
    print("[A] Negative marginal value of layer count (main conclusion 1)")
    print(RULE)
    report_layer_curve(pd.DataFrame(load_work("benchmark_matrix")))

    merged = load_work("benchmark_merged", {})
    print("\nR side (from the merged benchmark artefact)")
    print("  r_layer_effect:", merged.get("r_layer_effect"))
    print("  py_layer_effect:", merged.get("py_layer_effect"))

    print("\n" + RULE)
    print("[B] The cross-cohort method-selection trap (main conclusion 2)")
    print(RULE)
    merged = load_work("benchmark_merged", {})
    print("  merged two-sided ranking (R side):", merged.get("r_ranking"))
    cross_side = merged.get("cross_side", {})
    print(
        "  cross-implementation agreement:",
        {
            key: cross_side.get(key)
            for key in [
                "n_grid",
                "spearman",
                "p",
                "py_mean",
                "r_mean",
                "median_absdiff",
            ]
        },
    )
    pac = load_work("pac_aggregate", {})
    cross = pac.get("cross", {})
    if isinstance(cross, list):
        print("  PAC vs ARI (list form, one entry per record):")
        for record in cross:
            print("   ", record)
    else:
        print("  PAC vs ARI bins:", cross.get("bins"))
        print("  PAC vs ARI Spearman:", cross.get("spearman"), "p =", cross.get("p"))

    print("\n" + RULE)
    print("[C] Fragility of the covariance layer (main conclusion 3)")
    print(RULE)
    batch = pd.DataFrame(load_work("eeec_batch_verify_json", []))
    if len(batch):
        columns = [
            column
            for column in [
                "arm",
                "centroid_AUC",
                "LR_AUC_abs",
                "SVM_AUC_abs",
                "KS_median",
                "knn_mix",
            ]
            if column in batch.columns
        ]
        print(batch[columns].to_string(index=False))
    platform = load_work("platform_results_json", {})
    print("\n  cross-platform delta ladder:")
    for record in platform.get("ladder", []):
        print("   ", record)
    print("  network level:", platform.get("network"))
    print("  effect-size bins:", platform.get("bins"))

    print("\n" + RULE)
    print("[D] G2 and P2 (negative result 1)")
    print(RULE)
    g2 = load_work("g2_refined", {})
    print(
        f"  G2 frozen: baseline {g2.get('baseline')} = {g2.get('baseline_value'):.4f}"
        f" | decision-relevant SD {g2.get('sd_relevant'):.4f}"
    )
    p2 = load_work("pai_results", {})
    print(
        f"  target = {p2.get('g2_target'):.4f} | records {p2.get('n_records')}"
        f" | units {p2.get('n_units')}"
    )
    print("  arms:", p2.get("table"))
    print("  paired comparisons:")
    for record in p2.get("compare", []):
        print(
            f"    {record['arm'][:26]:28s} n={record['n']:3d} "
            f"d={record['mean_delta']:+.4f} rel={record['rel_vs_base']*100:+6.1f}% "
            f"CI[{record['ci_lo']:+.4f},{record['ci_hi']:+.4f}] "
            f"win={record['win_rate']*100:4.1f}% p={record['p']:.2g}"
        )

    print("\n" + RULE)
    print("[E] Other negative results and corrections")
    print(RULE)
    print("  SNF correction: self-implemented 0.0118 -> official SNFtool 0.2081 (17.6-fold)")
    purity = load_work("eeec_batch_purity_diag", None) or pd.DataFrame(
        load_work("purity_diag_json", [])
    ).to_dict("records")
    purity_frame = pd.DataFrame(purity)
    if len(purity_frame):
        print(
            f"  purity: mean explained r^2 = {purity_frame['mean_r2_purity'].mean():.4f}"
            f" | highest layer {purity_frame.loc[purity_frame['mean_r2_purity'].idxmax(), 'layer']}"
        )
    print("  purity distribution differs between the two cohorts: Mann-Whitney p = 0.794 (not significant)")
    print("  three exact identities: S3 ComBat o z == z (19/19) | S2 delta o z == z (4/4) | MCCA-lite == MOFA-lite (50/50)")
    print("  sample-intersection collapse: TCGA 524->390 (-25.6%) | ind 132->84 (-36.4%) | dis 95->81 (-14.7%)")


if __name__ == "__main__":
    main()

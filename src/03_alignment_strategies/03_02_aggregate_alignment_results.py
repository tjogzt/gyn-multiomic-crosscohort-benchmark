#!/usr/bin/env python3
"""
Aggregate the transfer-ARI matrix of the alignment-strategy sweep.

Purpose
    The alignment sweep writes one record per (layer, cohort pair, strategy), and
    the same (layer, pair, strategy) stratum can appear more than once when a
    sweep was extended. This module deduplicates the matrix, verifies that the
    three non-baseline arms really do change the result by comparing each of them
    with the S0 baseline value by value, and then aggregates the primary K:
    a layer-by-strategy table of mean transfer ARI, a per-pair detail table, the
    direction in which S1 quantile normalisation moves each pair, and the
    per-layer reproducibility ceiling of the baseline arm. Together these are the
    numbers that decide gate G4 for the alignment strategies.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("alignment_ari")             .json, records of stage 03 part 1. Records
        without a strategy field are skipped pairs and are counted, not scored.

Outputs
    work("alignment_ari_aggregate")   .json with the keys "agg_K4" (layer by
        strategy means at the primary K; the key name is kept for artefact
        compatibility), "detail" (per layer-pair rows) and "direction" (S1 versus
        S0 counts of improved, worsened and unchanged pairs).

Usage
    python 03_02_aggregate_alignment_results.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import ensure_dir, param, work  # noqa: E402

# Placeholder printed for a stratum that carries no value.
MISSING = "-"


def main() -> None:
    """Deduplicate, verify and aggregate the transfer-ARI matrix."""
    records = json.loads(work("alignment_ari").read_text(encoding="utf-8"))
    strategies = param("alignment", "strategies")
    baseline, quantile = strategies[0], strategies[1]
    layer_order = list(param("alignment", "layer_cohorts").keys())
    k_values = param("clustering", "k_values")
    k_primary = str(param("clustering", "k_primary"))
    tolerance = param("alignment", "ari_direction_tolerance")

    # A stratum is unique per (layer, pair, strategy); a later record replaces an
    # earlier one, so a re-run of a subset of pairs can extend an existing matrix.
    unique: dict[tuple, dict] = {}
    for record in records:
        if not record.get("strategy"):
            continue
        unique[(record["layer"], record["pair"], record["strategy"])] = record
    print(
        "unique records: %d (read %d, deduplicated %d)"
        % (len(unique), len(records), len(records) - len(unique))
    )

    strata = sorted({(key[0], key[1]) for key in unique})
    print("unique (layer, pair) strata: %d" % len(strata))

    print("\n" + "=" * 90)
    print(
        "[A] Identity check: largest per-value difference between each arm and the"
        " baseline (K=%s)" % "/".join(str(k) for k in k_values)
    )
    print("=" * 90)
    for strategy in strategies[1:]:
        differences, comparable = [], 0
        for layer, pair in strata:
            record_a = unique.get((layer, pair, baseline))
            record_b = unique.get((layer, pair, strategy))
            # A stratum is comparable only when both arms have the first grid
            # value; the maximum is then taken over the whole K grid.
            if (
                record_a
                and record_b
                and record_a.get("ari", {}).get(str(k_values[0])) is not None
                and record_b.get("ari", {}).get(str(k_values[0])) is not None
            ):
                differences.append(
                    max(
                        abs(
                            record_a["ari"][str(k)] - record_b["ari"][str(k)]
                        )
                        for k in k_values
                    )
                )
                comparable += 1
        if comparable:
            print(
                "  %-18s comparable %2d pairs | largest per-value difference %.6f"
                " | identical %d/%d"
                % (
                    strategy,
                    comparable,
                    max(differences),
                    sum(1 for value in differences if value < 1e-9),
                    comparable,
                )
            )

    print("\n" + "=" * 90)
    print(
        "[B] Layer x strategy: transfer ARI (K=%s, mean over cohort pairs)"
        % k_primary
    )
    print("=" * 90)
    aggregated: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for layer, pair in strata:
        for strategy in strategies:
            record = unique.get((layer, pair, strategy))
            if (
                record
                and record.get("ari")
                and record["ari"].get(k_primary) is not None
            ):
                aggregated[layer][strategy].append(record["ari"][k_primary])
    print(
        "%-9s%5s  " % ("layer", "pairs")
        + "".join("%18s" % strategy for strategy in strategies)
    )
    table: dict[str, dict] = {}
    for layer in layer_order:
        n_pairs = len({pair for (layer_name, pair) in strata if layer_name == layer})
        row: dict[str, float | None] = {}
        line = "%-9s%5d  " % (layer, n_pairs)
        for strategy in strategies:
            values = aggregated[layer].get(strategy)
            if values:
                row[strategy] = round(float(np.mean(values)), 4)
                line += "%18.4f" % np.mean(values)
            else:
                row[strategy] = None
                line += "%18s" % MISSING
        table[layer] = row
        print(line)

    print("\n" + "=" * 90)
    print("[C] Per cohort-pair detail (K=%s)" % k_primary)
    print("=" * 90)
    print(
        "%-9s%-14s%11s%7s  " % ("layer", "pair", "nA/nB", "genes")
        + "".join("%14s" % strategy[:12] for strategy in strategies)
    )
    detail: list[dict] = []
    for layer, pair in strata:
        baseline_record = unique.get((layer, pair, baseline))
        row = {
            "layer": layer,
            "pair": pair,
            "nA": baseline_record["nA"] if baseline_record else None,
            "nB": baseline_record["nB"] if baseline_record else None,
            "ngene": baseline_record["ngene"] if baseline_record else None,
        }
        line = "%-9s%-14s%11s%7d  " % (
            layer,
            pair,
            (
                "%s/%s" % (row["nA"], row["nB"])
                if baseline_record
                else MISSING
            ),
            row["ngene"] or 0,
        )
        for strategy in strategies:
            record = unique.get((layer, pair, strategy))
            value = (
                record["ari"][k_primary]
                if (
                    record
                    and record.get("ari")
                    and record["ari"].get(k_primary) is not None
                )
                else None
            )
            row[strategy] = value
            line += "%14s" % ("%.4f" % value if value is not None else MISSING)
        detail.append(row)
        print(line)

    print("\n" + "=" * 90)
    print(
        "[D] Direction of the quantile-normalisation arm relative to the baseline"
        " (K=%s)" % k_primary
    )
    print("=" * 90)
    better = worse = same = 0
    for row in detail:
        value_baseline, value_quantile = row[baseline], row[quantile]
        if value_baseline is None or value_quantile is None:
            continue
        if value_quantile > value_baseline + tolerance:
            tag = "improved"
            better += 1
        elif value_quantile < value_baseline - tolerance:
            tag = "worsened"
            worse += 1
        else:
            tag = "unchanged"
            same += 1
        print(
            "  %-9s%-14s%.4f -> %.4f   %s"
            % (row["layer"], row["pair"], value_baseline, value_quantile, tag)
        )
    print(
        "  -> total: improved %d / worsened %d / unchanged %d  (%d pairs)"
        % (better, worse, same, better + worse + same)
    )

    print("\n" + "=" * 90)
    print("[E] Cross-cohort reproducibility ceiling (baseline arm, K=%s)" % k_primary)
    print("=" * 90)
    for layer in layer_order:
        values = sorted(
            [
                row[baseline]
                for row in detail
                if row["layer"] == layer and row[baseline] is not None
            ],
            reverse=True,
        )
        if values:
            print(
                "  %-9s highest %.4f | lowest %.4f | mean %.4f | range %.4f"
                % (layer, values[0], values[-1], np.mean(values), values[0] - values[-1])
            )

    out_path = work("alignment_ari_aggregate")
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "agg_K4": table,
                "detail": detail,
                "direction": {"better": better, "worse": worse, "same": same},
            },
            handle,
            indent=1,
            ensure_ascii=False,
        )
    print("\nWrote the alignment aggregate to %s" % out_path)


if __name__ == "__main__":
    main()

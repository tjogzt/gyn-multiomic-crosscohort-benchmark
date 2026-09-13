#!/usr/bin/env python3
"""
Aggregate the PAC records and cross them with the cross-cohort transfer ARI.

Purpose
    Stage 06, second step. It reads the per-arm stability records written by
    06_01 and the cross-cohort alignment records written by the Python benchmark
    (work.benchmark_matrix), and answers the question the stability analysis
    exists for: is a more stable within-cohort clustering also a more
    transferable one? The script prints three blocks - method x PAC, layer subset
    x PAC, and PAC x transfer ARI including the per-domain correlations and the
    PAC-bin summary - and writes the method table together with the individual
    cross records to work.pac_aggregate for the result tables and the figures.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work.pac_matrix
        JSON list of per-arm stability records from 06_01; records with a null
        pac are dropped.
    work.benchmark_matrix
        JSON list of cross-cohort alignment records; each record carries domain,
        subset, pair ("<cohort A>x<cohort B>"), method and ari keyed by k as a
        string.
    params clustering.k_primary (the k at which transfer ARI is read),
    params stability.pac_bins, stability.min_cross_records.

Outputs
    work.pac_aggregate
        JSON object with tab1 (per method: mean PAC by domain, pooled mean PAC,
        mean strong-consensus fraction, mean median consensus, record count) and
        cross (the individual PAC x ARI cross records).

Usage
    python src/06_consensus_stability/06_02_aggregate_pac_results.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# The package root (one level above src/) so that "common" can be imported from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import ensure_dir, param, work

# Separator inside a cohort-pair label; declared in params.yaml because
# downstream stages split records on it. See the note there.
PAIR_SEPARATOR = param("benchmark", "pair_separator")

# --- Configuration -----------------------------------------------------------

# Transfer ARI is read at the primary k reported by the benchmark.
K = str(param("clustering", "k_primary"))

# PAC bins used to read transfer ARI as a function of stability.
PAC_BINS = [tuple(float(bound) for bound in pair) for pair in param("stability", "pac_bins")]

# A per-domain correlation is only reported from this many cross records upwards.
MIN_CROSS_RECORDS = int(param("stability", "min_cross_records"))

# Domain labels, identical to the "domain" values 04_02 and 04_03 write into
# work.benchmark_matrix: the cross records are joined on them, so they are never
# printed in another form here.
DOMAIN = param("domains")
DOMAIN_LABELS = [DOMAIN["A"], DOMAIN["B"]]

PAC_MATRIX = work("pac_matrix")
BENCHMARK_MATRIX = work("benchmark_matrix")
OUT_PATH = work("pac_aggregate")


def load_records() -> tuple[list, list, list]:
    """Read the stability and benchmark records and apply the validity filters.

    Returns:
        A triple (stability records with a PAC, benchmark records with an ARI at
        the primary k, the same records with the domain label and the transfer
        ARI extracted for readability).
    """
    stability = json.loads(PAC_MATRIX.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_MATRIX.read_text(encoding="utf-8"))
    scored = [record for record in stability if record.get("pac") is not None]
    aligned = [
        record
        for record in benchmark
        if record.get("ari") and record["ari"].get(K) is not None
    ]
    return stability, scored, aligned


def report_method_table(scored: list) -> dict:
    """Print and build the method x PAC table.

    Args:
        scored: stability records that carry a PAC value.

    Returns:
        Mapping method label -> mean PAC across all records, mean PAC per domain,
        mean strong-consensus fraction, mean median consensus and record count.
    """
    print("=" * 104)
    print(f"[1] Method x PAC (K={K}, within-cohort stability; lower PAC is more stable)")
    print("=" * 104)
    methods = sorted({record["method"] for record in scored})
    print(
        f"{'method':<30}{'PAC dom A':>10}{'PAC dom B':>10}{'PAC pooled':>11}"
        f"{'strong':>9}{'median':>9}{'n':>5}"
    )
    table = {}
    for method in methods:
        rows = [record for record in scored if record["method"] == method]
        # Mean PAC per domain, then the pooled mean over every arm of the method.
        domain_a = [r["pac"] for r in rows if r["domain"] == DOMAIN_LABELS[0]]
        domain_b = [r["pac"] for r in rows if r["domain"] == DOMAIN_LABELS[1]]
        table[method] = {
            "A": round(float(np.mean(domain_a)), 4),
            "B": round(float(np.mean(domain_b)), 4),
            "all": round(float(np.mean([r["pac"] for r in rows])), 4),
            "strong": round(float(np.mean([r["strong"] for r in rows])), 3),
            "med": round(float(np.mean([r["median_cons"] for r in rows])), 3),
            "n": len(rows),
        }
        # The numeric prefix of the label ("1_SNF" -> "SNF") is not repeated in
        # the printed name; the ranking line below carries the prefix-free name.
        print(
            f"{method.split('_', 1)[1]:<30}{table[method]['A']:>10.4f}"
            f"{table[method]['B']:>10.4f}{table[method]['all']:>11.4f}"
            f"{table[method]['strong']:>9.3f}{table[method]['med']:>9.3f}"
            f"{table[method]['n']:>5}"
        )
    ranking = sorted(methods, key=lambda method: table[method]["all"])
    print(
        "\n  Most stable -> least stable: "
        + " < ".join(f"{m.split('_', 1)[1]}({table[m]['all']:.3f})" for m in ranking)
    )
    return table


def report_subset_table(scored: list) -> None:
    """Print the layer subset x PAC summary, grouped by number of layers.

    Args:
        scored: stability records that carry a PAC value.
    """
    print("\n" + "=" * 104)
    print("[2] Layer subset x PAC (averaged across cohorts, grouped by layer count)")
    print("=" * 104)
    for domain in DOMAIN_LABELS:
        rows = [record for record in scored if record["domain"] == domain]
        subsets = sorted(
            {record["subset"] for record in rows},
            key=lambda subset: (len(subset.split("+")), subset),
        )
        print(f"\n  {domain}")
        for size in sorted({len(subset.split("+")) for subset in subsets}):
            at_size = [s for s in subsets if len(s.split("+")) == size]
            values = [record["pac"] for record in rows if record["subset"] in at_size]
            print(
                f"    {size} layer(s) | subsets {len(at_size):2d} | mean PAC {np.mean(values):.4f}"
                f" | most stable {min(values):.4f} | least stable {max(values):.4f}"
            )


def build_cross_records(scored: list, aligned: list) -> list:
    """Join every benchmark record with the PAC of both of its cohorts.

    Args:
        scored: stability records that carry a PAC value, keyed by
            (domain, cohort, subset, method).
        aligned: benchmark records that carry an ARI at the primary k.

    Returns:
        One record per benchmark entry whose two cohorts both have a PAC for the
        same domain, layer subset and method; the PAC of the pair is the mean of
        the two within-cohort PAC values.
    """
    pac_by_arm = {
        (record["domain"], record["cohort"], record["subset"], record["method"]): record["pac"]
        for record in scored
    }
    cross = []
    for record in aligned:
        domain, subset, pair, method = (
            record["domain"],
            record["subset"],
            record["pair"],
            record["method"],
        )
        # Split the pair label on the configured separator (params.yaml ->
        # benchmark.pair_separator). The label is written by the benchmark
        # stage as "<cohort A><separator><cohort B>".
        cohort_a, cohort_b = pair.split(PAIR_SEPARATOR)
        pac_a = pac_by_arm.get((domain, cohort_a, subset, method))
        pac_b = pac_by_arm.get((domain, cohort_b, subset, method))
        if pac_a is not None and pac_b is not None:
            cross.append(
                {
                    "domain": domain,
                    "subset": subset,
                    "pair": pair,
                    "method": method,
                    "ari": record["ari"][K],
                    "pac": (pac_a + pac_b) / 2,
                    "pacA": pac_a,
                    "pacB": pac_b,
                    "nA": record["nA"],
                    "nB": record["nB"],
                }
            )
    return cross


def report_cross_table(cross: list) -> None:
    """Print the PAC x transfer-ARI cross analysis.

    Args:
        cross: PAC x ARI cross records built by build_cross_records.
    """
    from scipy.stats import pearsonr, spearmanr

    print("\n" + "=" * 104)
    print("[3] PAC x cross-cohort ARI (core question: is a more stable arm also more transferable?)")
    print("=" * 104)
    print(f"  cross records available: {len(cross)}")
    if not cross:
        return
    ari = np.array([record["ari"] for record in cross])
    pac = np.array([record["pac"] for record in cross])
    # A negative correlation means stability travels with transferability; a
    # positive one means the two properties are unrelated or opposed.
    print(
        f"  Spearman(ARI, PAC) = {spearmanr(ari, pac)[0]:+.4f} "
        f"(p={spearmanr(ari, pac)[1]:.2e})"
    )
    print(
        f"  Pearson (ARI, PAC) = {pearsonr(ari, pac)[0]:+.4f} "
        f"(p={pearsonr(ari, pac)[1]:.2e})"
    )
    print("  negative = more stable is more transferable; positive = no or reverse relation")
    for domain in DOMAIN_LABELS:
        subset = [record for record in cross if record["domain"] == domain]
        if len(subset) > MIN_CROSS_RECORDS:
            ari_d = np.array([record["ari"] for record in subset])
            pac_d = np.array([record["pac"] for record in subset])
            print(
                f"      {domain}: n={len(subset)}  Spearman {spearmanr(ari_d, pac_d)[0]:+.4f} "
                f"(p={spearmanr(ari_d, pac_d)[1]:.2e})"
            )
    print("\n  Transfer ARI by PAC bin")
    for low, high in PAC_BINS:
        subset = [record for record in cross if low <= record["pac"] < high]
        if subset:
            print(
                f"      PAC [{low:.2f},{high:.2f}) n={len(subset):3d}  "
                f"mean ARI {np.mean([record['ari'] for record in subset]):.4f}"
            )
    print("\n  Most and least stable layer subsets (by PAC)")
    for domain in DOMAIN_LABELS:
        subset = [record for record in cross if record["domain"] == domain]
        if not subset:
            continue
        # Accumulate the mean PAC of each layer subset, then order the subsets.
        accumulated = defaultdict(lambda: [0.0, 0])
        for record in subset:
            accumulated[record["subset"]][0] += record["pac"]
            accumulated[record["subset"]][1] += 1
        ordered = sorted(accumulated, key=lambda key: accumulated[key][0] / accumulated[key][1])
        best, worst = ordered[0], ordered[-1]
        print(
            f"      {domain}: most stable {best} "
            f"(PAC {accumulated[best][0] / accumulated[best][1]:.3f}) | least stable {worst} "
            f"(PAC {accumulated[worst][0] / accumulated[worst][1]:.3f})"
        )


def main() -> dict:
    """Run the three report blocks and write work.pac_aggregate.

    Returns:
        The payload that was written to work.pac_aggregate.
    """
    _, scored, aligned = load_records()
    method_table = report_method_table(scored)
    report_subset_table(scored)
    cross = build_cross_records(scored, aligned)
    report_cross_table(cross)
    ensure_dir(OUT_PATH.parent)
    payload = {"tab1": method_table, "cross": cross}
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print(f"\nWritten to {OUT_PATH}")
    return payload


if __name__ == "__main__":
    main()

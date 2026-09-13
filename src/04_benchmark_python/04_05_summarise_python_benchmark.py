"""
Build the single summary table of the Python integration benchmark.

Purpose
    Consolidate the benchmark records into the four reporting tables quoted by
    the manuscript: (1) method means by domain and pooled, with the balance and
    degeneracy diagnostics; (2) layers -> best ARI together with the usable
    sample count; (3) the layer curve of the fixed best method; (4) layers ->
    usable samples per cohort. All means are record-weighted, i.e. an equal-weight
    mean over every (layer subset x cohort pair) record. The module reads the
    record and availability files and writes ``work("benchmark_summary")``; no
    value is recomputed from raw data here.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")        JSON benchmark records.
    work("benchmark_availability")  JSON sample-availability records.
    param("clustering", "k_primary")            The K reported as the main result.
    param("domains")                            Evaluation domains and labels.
    param("benchmark", "best_python_method")    Arm reported as the fixed best.
    param("cohort_tags")                        Frozen cohort tags for table 4.

Outputs
    work("benchmark_summary")  JSON object with key "tab1".."tab4".

Usage
    python 04_05_summarise_python_benchmark.py
"""

import sys
from pathlib import Path

# Make the shared configuration loader importable regardless of the caller's
# working directory: this file sits in src/<stage>/, so ``parents[1]`` is src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import work, param

import json
from collections import defaultdict

import numpy as np

# --- Configuration -----------------------------------------------------------
MATRIX_FILE = work("benchmark_matrix")
AVAILABILITY_FILE = work("benchmark_availability")
SUMMARY_FILE = work("benchmark_summary")

# JSON object keys are strings, so the reported K is looked up as a string.
K = str(param("clustering", "k_primary"))
DOMAIN = param("domains")
# The fixed best-performing Python arm whose layer curve is reported.
BEST_METHOD = param("benchmark", "best_python_method")
COHORT = param("cohort_tags")

RECORDS = json.load(open(MATRIX_FILE))
AVAILABILITY = json.load(open(AVAILABILITY_FILE))
OK = [x for x in RECORDS if x.get("ari") and x["ari"].get(K) is not None]
METHODS = sorted({x["method"] for x in OK})


def build_table1():
    """Build the method x domain table and print the pooled ranking.

    Returns:
        ``(table, ranking)`` where ``table`` maps a method label to its domain-A,
        domain-B and pooled means, its largest-cluster fraction, its degeneracy
        rate and its record count, and ``ranking`` is the methods ordered by
        pooled mean ARI.
    """
    print("Convention: a method/subset mean is the equal-weight mean over all")
    print("(layer subset x cohort pair) records\n")
    print("=" * 96)
    print("Table 1  method x domain (record-weighted transfer ARI, K=%s)" % K)
    print("=" * 96)
    print(
        f"{'method':<20}{'domainA':>10}{'domainB':>10}{'pooled':>10}"
        f"{'maxCluster':>12}{'degeneracy':>12}{'n':>5}"
    )
    table = {}
    for m in METHODS:
        r = [x for x in OK if x["method"] == m]
        a = [x["ari"][K] for x in r if x["domain"] == DOMAIN["A"]]
        b = [x["ari"][K] for x in r if x["domain"] == DOMAIN["B"]]
        bal = np.mean(
            [max(x["balance"][K]) for x in r if x.get("balance") and x["balance"].get(K)]
        )
        dg = np.mean(
            [x["degen"][K] for x in r if x.get("degen") and x["degen"].get(K) is not None]
        )
        table[m] = {
            "A": round(float(np.mean(a)), 4),
            "B": round(float(np.mean(b)), 4),
            "all": round(float(np.mean([x["ari"][K] for x in r])), 4),
            "bal": round(float(bal), 3),
            "deg": round(float(dg), 3),
            "n": len(r),
        }
        print(
            f"{m.split('_', 1)[1]:<20}{table[m]['A']:>10.4f}{table[m]['B']:>10.4f}"
            f"{table[m]['all']:>10.4f}{table[m]['bal']:>12.3f}"
            f"{table[m]['deg']:>12.2f}{table[m]['n']:>5}"
        )
    ranking = sorted(METHODS, key=lambda m: -table[m]["all"])
    print(
        "\nranking (pooled ARI): "
        + " > ".join(f"{m.split('_', 1)[1]}({table[m]['all']:.3f})" for m in ranking)
    )
    return table, ranking


def build_table2():
    """Build the layers -> best ARI table for each domain.

    The best entry at a layer count is the (subset, method) cell with the highest
    record-weighted mean ARI at that count.

    Returns:
        A dict mapping a domain label to the list of per-layer-count records
        ``{"nL", "nsub", "best", "subset", "method"}``.
    """
    print("\n" + "=" * 96)
    print("Table 2  layers -> best ARI (record maximum) and usable samples")
    print("=" * 96)
    table = {}
    for dom in DOMAIN.values():
        r = [x for x in OK if x["domain"] == dom]
        subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
        print(f"\n|{dom}")
        print(f"{'nLayers':<8}{'nSubsets':>9}{'bestARI':>10}   best subset x method")
        table[dom] = []
        for n in sorted({len(s.split("+")) for s in subs}):
            ss = [s for s in subs if len(s.split("+")) == n]
            best = None
            for s in ss:
                for m in METHODS:
                    v = [x["ari"][K] for x in r if x["subset"] == s and x["method"] == m]
                    if v and (best is None or np.mean(v) > best[2]):
                        best = (s, m, float(np.mean(v)))
            print(f"{n:<8}{len(ss):>9}{best[2]:>10.4f}   {best[0]} x {best[1].split('_', 1)[1]}")
            table[dom].append(
                {
                    "nL": n,
                    "nsub": len(ss),
                    "best": round(best[2], 4),
                    "subset": best[0],
                    "method": best[1].split("_", 1)[1],
                }
            )
    return table


def build_table3():
    """Build the layer curve of the fixed best method, per domain.

    Returns:
        A dict mapping a domain label to the list of per-layer-count records
        ``{"nL", "n", "ari", "max", "nA"}`` for the fixed best method.
    """
    print("\n" + "=" * 96)
    print(f"Table 3  layer curve of the fixed best method {BEST_METHOD.split('_', 1)[1]}")
    print("=" * 96)
    table = {}
    for dom in DOMAIN.values():
        r = [x for x in OK if x["domain"] == dom and x["method"] == BEST_METHOD]
        print(f"\n|{dom}")
        table[dom] = []
        for n in sorted({len(x["subset"].split("+")) for x in r}):
            v = [x["ari"][K] for x in r if len(x["subset"].split("+")) == n]
            na = [x["nA"] for x in r if len(x["subset"].split("+")) == n]
            print(
                f"   {n} layers | pairs {len(v):2d} | ARI {np.mean(v):.4f} | "
                f"max {max(v):.4f} | mean samples {int(np.mean(na))}"
            )
            table[dom].append(
                {
                    "nL": n,
                    "n": len(v),
                    "ari": round(float(np.mean(v)), 4),
                    "max": round(float(max(v)), 4),
                    "nA": int(np.mean(na)),
                }
            )
    return table


def build_table4():
    """Build the layers -> usable-samples table, per cohort.

    Returns:
        A nested dict ``{cohort: {"<nLayers>|<subset>": n_samples}}`` restricted
        to the cohorts that actually appear in the availability file.
    """
    print("\n" + "=" * 96)
    print("Table 4  layers -> usable samples (per cohort, layer intersection)")
    print("=" * 96)
    table = defaultdict(dict)
    for x in AVAILABILITY:
        table[x["cohort"]][(len(x["subset"].split("+")), x["subset"])] = x["n"]
    # Iterate in the frozen cohort order of the domains.
    cohort_order = [COHORT["tcga"], COHORT["discovery"], COHORT["independent"], COHORT["ov"]]
    for coh in cohort_order:
        if coh not in table:
            continue
        print(f"\n  |{coh}")
        for (n, s), v in sorted(table[coh].items()):
            print(f"      {n} layers  {s:<28} n={v}")
    return table


def main():
    """Build the four tables and write the combined summary file."""
    tab1, _ranking = build_table1()
    tab2 = build_table2()
    tab3 = build_table3()
    tab4 = build_table4()
    # Table-4 keys are tuples, which JSON cannot represent; flatten them to
    # "layers|subset" strings, which is how the table is quoted.
    payload = {
        "tab1": tab1,
        "tab2": tab2,
        "tab3": tab3,
        "tab4": {k: {f"{a}|{b}": c for (a, b), c in v.items()} for k, v in tab4.items()},
    }
    with open(SUMMARY_FILE, "w") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print("\nwritten:", SUMMARY_FILE)


if __name__ == "__main__":
    main()

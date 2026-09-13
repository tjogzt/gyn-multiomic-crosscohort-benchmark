"""
Aggregate the Python integration benchmark records into per-subset summaries.

Purpose
    Reduce ``work("benchmark_matrix")`` to the layer-subset x method table of
    mean transfer ARI, the per-method mean across subsets, and the per-layer-count
    marginal-value curve, and write the best-method table for each evaluation
    domain. It reads the record files produced by the Python benchmark driver and
    the availability file, and writes one ``bench_best_<domain>.json`` per domain.
    It is a reporting stage: no value is recomputed from raw data.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")        JSON benchmark records.
    work("benchmark_availability")  JSON sample-availability records.
    param("clustering", "k_primary")  The K reported as the main result.
    param("domains")                Evaluation domains and their labels.

Outputs
    work("benchmark_best_pattern")  One JSON file per domain mapping each layer
                                    subset to its best method and best mean ARI.

Usage
    python 04_03_aggregate_python_results.py
"""

import sys
from pathlib import Path

# Make the shared configuration loader importable regardless of the caller's
# working directory: this file sits in src/<stage>/, so ``parents[1]`` is src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import work, param

import json
from collections import Counter, defaultdict

import numpy as np

# --- Configuration -----------------------------------------------------------
MATRIX_FILE = work("benchmark_matrix")
AVAILABILITY_FILE = work("benchmark_availability")
BEST_PATTERN = str(work("benchmark_best_pattern"))

# JSON object keys are strings, so the reported K is looked up as a string.
K = str(param("clustering", "k_primary"))
DOMAINS = param("domains")
DOMAIN_KEY_BY_LABEL = {label: key for key, label in DOMAINS.items()}


def report_coverage(records):
    """Print how many records were computed, usable and in error, per domain.

    Args:
        records: The list of benchmark records.

    Returns:
        None. Prints a per-domain coverage line and the most frequent errors,
        which is how an incomplete run is spotted before the aggregate is read.
    """
    print("=" * 100)
    print("Coverage")
    print("=" * 100)
    for dom in sorted({x["domain"] for x in records}):
        r = [x for x in records if x["domain"] == dom]
        ok = [x for x in r if x.get("ari") and x["ari"].get(K) is not None]
        err = [x for x in r if x.get("err")]
        ec = Counter(x["err"][:34] for x in err)
        print(f"  {dom}: records {len(r)} | valid {len(ok)} | errors {len(err)}")
        for k, v in ec.most_common(4):
            print(f"      {v:3d}x {k}")


def summarise_domains(records):
    """Print and persist the per-subset x method table for every domain.

    Args:
        records: The list of benchmark records.

    Returns:
        None. Prints the table, the per-method means and the marginal-value
        curve, and writes one ``bench_best_<domain>.json`` per domain.
    """
    print("\n" + "=" * 100)
    print(f"Layer subset x method transfer ARI (K={K})")
    print("=" * 100)
    for dom in DOMAINS.values():
        r = [x for x in records if x["domain"] == dom and x.get("ari") and x["ari"].get(K) is not None]
        if not r:
            continue
        subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
        meths = sorted({x["method"] for x in r})
        agg = defaultdict(list)
        for x in r:
            agg[(x["subset"], x["method"])].append(x["ari"][K])
        print(f"\n|{dom}")
        print(
            f"{'layer subset':<26}{'nLayers':>7}{'nPairs':>7}  "
            + "".join(f"{m.split('_')[1][:9]:>11}" for m in meths)
            + f"{'best':>14}"
        )
        BEST = {}
        for s in subs:
            vals = {m: (np.mean(agg[(s, m)]) if agg.get((s, m)) else None) for m in meths}
            npair = max(len(agg[(s, m)]) for m in meths)
            line = f"{s:<26}{len(s.split('+')):>7}{npair:>7}  "
            for m in meths:
                v = vals[m]
                line += f"{(f'{v:.3f}' if v is not None else '-'):>11}"
            ok = {m: v for m, v in vals.items() if v is not None}
            bm = max(ok, key=ok.get) if ok else None
            line += f"{(bm.split('_')[1][:10] + ' ' + f'{ok[bm]:.3f}') if bm else '-':>14}"
            BEST[s] = (bm, ok.get(bm) if bm else None)
            print(line)
        # Per-method mean across all subsets of this domain.
        print("\n  > method means (across all subsets)")
        for m in meths:
            vs = [agg[(s, m)] for s in subs if agg.get((s, m))]
            if vs:
                print(f"      {m:<20} {np.mean([np.mean(v) for v in vs]):.4f}  (subsets {len(vs)})")
        # Marginal value: how the best achievable ARI grows with the layer count.
        print("\n  > marginal value: layers -> best ARI")
        for nl in sorted({len(s.split('+')) for s in subs}):
            ss = [s for s in subs if len(s.split('+')) == nl]
            vs = [BEST[s][1] for s in ss if BEST[s][1] is not None]
            if vs:
                bs = max(ss, key=lambda s: BEST[s][1] if BEST[s][1] is not None else -1)
                print(
                    f"      {nl} layers | subsets {len(ss):2d} | best ARI {max(vs):.4f} "
                    f"| best subset {bs}"
                )
        out_path = Path(BEST_PATTERN.format(domain=DOMAIN_KEY_BY_LABEL[dom]))
        with open(out_path, "w") as handle:
            json.dump(
                {s: {"best_method": BEST[s][0], "best_ari": BEST[s][1]} for s in subs},
                handle,
                indent=1,
                ensure_ascii=False,
            )
        print("  written:", out_path)


def main():
    """Load the benchmark records and run the coverage and subset summaries."""
    with open(MATRIX_FILE) as handle:
        records = json.load(handle)
    with open(AVAILABILITY_FILE) as handle:
        _availability = json.load(handle)  # loaded to assert the file is present
    report_coverage(records)
    summarise_domains(records)


if __name__ == "__main__":
    main()

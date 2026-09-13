"""
Final aggregation of the Python integration benchmark records.

Purpose
    Produce the refined per-method and per-subset tables of the Python benchmark:
    the pooled method ranking, the near-identity diagnostics that show which arms
    return the same partition on the same records, the domain-by-domain
    layer-subset x method mean table, and the per-layer marginal contribution of
    the best method. It reads the record files written by the Python benchmark
    driver and writes one ``bench_table_<domain>.json`` per domain. No value is
    recomputed from raw data here.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")        JSON benchmark records (with the balance and
                                    degeneracy fields written by the driver).
    work("benchmark_availability")  JSON sample-availability records.
    param("clustering", "k_primary")  The K reported as the main result.
    param("domains")                Evaluation domains and their labels.

Outputs
    work("benchmark_table_pattern")  One JSON file per domain holding the full
                                     layer-subset x method matrix of mean ARI.

Usage
    python 04_04_aggregate_python_results_refined.py
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
TABLE_PATTERN = str(work("benchmark_table_pattern"))

# JSON object keys are strings, so the reported K is looked up as a string.
K = str(param("clustering", "k_primary"))
DOMAINS = param("domains")
DOMAIN_KEY_BY_LABEL = {label: key for key, label in DOMAINS.items()}

# A record is usable only when it carries an ARI at the reported K; failed or
# degenerate records have ``ari`` empty or missing.
RECORDS = json.load(open(MATRIX_FILE))
OK = [x for x in RECORDS if x.get("ari") and x["ari"].get(K) is not None]


def report_method_ranking():
    """Print the pooled method ranking with its balance and degeneracy diagnostics.

    Returns:
        None. Prints one line per method.
    """
    print("=" * 104)
    print("[1] Overall method performance (K=%s, domains A and B pooled)" % K)
    print("=" * 104)
    for m in sorted({x["method"] for x in OK}):
        r = [x for x in OK if x["method"] == m]
        ari = np.mean([x["ari"][K] for x in r])
        bal = np.mean([max(x["balance"][K]) for x in r if x.get("balance") and x["balance"].get(K)])
        dg = np.mean(
            [x["degen"][K] for x in r if x.get("degen") and x["degen"].get(K) is not None]
        )
        print(
            f"  {m:<20} mean ARI {ari:.4f} | largest cluster {bal:.3f} | "
            f"degeneracy {dg:.2f} | n={len(r)}"
        )


def report_method_identity():
    """Print the near-identity comparison between methods, record by record.

    Two methods are compared only over the records they share; a pair compared on
    fewer than five records is skipped as uninformative.

    Returns:
        None. Prints one line per compared method pair.
    """
    print("\n" + "=" * 104)
    print("[2] Identical / near-identical method pairs (record-by-record)")
    print("=" * 104)
    ms = sorted({x["method"] for x in OK})
    key = lambda x: (x["domain"], x["subset"], x["pair"])
    by = defaultdict(dict)
    for x in OK:
        by[key(x)][x["method"]] = x["ari"][K]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            d = [abs(v[ms[i]] - v[ms[j]]) for v in by.values() if ms[i] in v and ms[j] in v]
            # Five shared records is the minimum for an informative comparison.
            if len(d) >= 5:
                same = sum(1 for t in d if t < 1e-9)
                print(
                    f"  {ms[i]:<20} vs {ms[j]:<20} comparable {len(d):3d} | "
                    f"max diff {max(d):.4f} | identical {same}/{len(d)}"
                )


def summarise_domains():
    """Print and persist the layer-subset x method table for every domain.

    Returns:
        None. Prints the table, the per-method means and the marginal-value
        curve, and writes one ``bench_table_<domain>.json`` per domain.
    """
    for dom in DOMAINS.values():
        r = [x for x in OK if x["domain"] == dom]
        if not r:
            continue
        subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
        meths = sorted({x["method"] for x in r})
        print(f"\n{'=' * 104}\n[3] {dom} - layer subset x method (K={K}, mean over cohort pairs)\n{'=' * 104}")
        print(
            f"{'layer subset':<26}{'nL':>3}{'nPairs':>7}  "
            + "".join(f"{m.split('_')[1][:8]:>10}" for m in meths)
        )
        agg = defaultdict(list)
        keep = {}
        for s in subs:
            line = f"{s:<26}{len(s.split('+')):>3}"
            npair = len({x["pair"] for x in r if x["subset"] == s})
            line += f"{npair:>7}  "
            row = {}
            for m in meths:
                v = [x["ari"][K] for x in r if x["subset"] == s and x["method"] == m]
                row[m] = round(float(np.mean(v)), 4) if v else None
                agg[m].append(row[m])
                line += f"{(f'{row[m]:.3f}' if row[m] is not None else '-'):>10}"
            keep[s] = row
            print(line)
        print("\n  > per-method mean (across subsets)")
        for m in meths:
            vs = [v for v in agg[m] if v is not None]
            if vs:
                print(f"      {m:<20} {np.mean(vs):.4f}")
        print("\n  > marginal value: layers -> best ARI / best subset")
        for nl in sorted({len(s.split('+')) for s in subs}):
            ss = [s for s in subs if len(s.split('+')) == nl]
            best = None
            for s in ss:
                for m, v in keep[s].items():
                    if v is not None and (best is None or v > best[2]):
                        best = (s, m, v)
            if best:
                print(
                    f"      {nl} layers | {len(ss):2d} subsets | best {best[2]:.4f}  "
                    f"<- {best[0]} x {best[1].split('_')[1]}"
                )
        out_path = Path(TABLE_PATTERN.format(domain=DOMAIN_KEY_BY_LABEL[dom]))
        with open(out_path, "w") as handle:
            json.dump(keep, handle, indent=1, ensure_ascii=False)
        print("  written:", out_path)


def report_marginal_contribution():
    """Print the per-layer curve of the single best method of each domain.

    Returns:
        None. Prints, for the best method of each domain, the mean ARI and mean
        sample count at every layer count.
    """
    print("\n" + "=" * 104)
    print("[4] Per-layer marginal contribution (net gain of adding a layer, K=%s)" % K)
    print("=" * 104)
    for dom in DOMAINS.values():
        r = [x for x in OK if x["domain"] == dom]
        if not r:
            continue
        meths = sorted({x["method"] for x in r})
        # The domain's best method is the one with the highest mean ARI overall.
        best_m = max(
            meths, key=lambda m: np.mean([x["ari"][K] for x in r if x["method"] == m])
        )
        print(f"\n  |{dom}  best method = {best_m}")
        subs = sorted({x["subset"] for x in r}, key=lambda s: (len(s.split("+")), s))
        for s in subs:
            v = [x["ari"][K] for x in r if x["subset"] == s and x["method"] == best_m]
            n = [x["nA"] for x in r if x["subset"] == s and x["method"] == best_m]
            if v:
                print(f"      {s:<26} ARI {np.mean(v):.4f} | mean nA {int(np.mean(n))}")


def main():
    """Load the benchmark records and run the four reporting sections."""
    with open(AVAILABILITY_FILE) as handle:
        _availability = json.load(handle)  # loaded to assert the file is present
    report_method_ranking()
    report_method_identity()
    summarise_domains()
    report_marginal_contribution()


if __name__ == "__main__":
    main()

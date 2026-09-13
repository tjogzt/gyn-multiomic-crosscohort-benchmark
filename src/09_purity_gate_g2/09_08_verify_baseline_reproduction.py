"""
Verify that the purity-aware baseline arm reproduces the benchmark baseline.

Purpose
    Closing check of the purity gate (G2). The gate decides the purity-aware arms
    against a frozen baseline transfer ARI of 0.4027, which was measured on the
    KMeans(concat) arm of the cross-cohort benchmark. That comparison is only
    valid if the B0 baseline arm of the purity run reproduces those benchmark
    records on exactly the same evaluation units (domain x layer subset x cohort
    pair). If the two baselines disagreed, a purity arm could appear to help or to
    hurt merely because it was scored on a different set of units, or on a unit
    where one side had already been filtered out as degenerate.
    The module joins the two record files on a normalised unit key, reports the
    difference per shared record, lists the records only one side holds, and
    compares the two means after degeneracy filtering on both sides. It writes no
    artefact: its evidence is the console report, and it is what justifies reading
    the arm comparison of 09_07 as a comparison of purity handling rather than of
    two different baselines.
    The baseline being reproduced is the measured quantity the gate threshold was
    derived from. The +46.6% threshold is not an assumed round number: it is the
    improvement the design was measured to resolve given the decision-relevant
    noise floor, so reproducing the baseline record by record is exactly what
    keeps that derivation valid.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")  JSON, record matrix of the cross-cohort benchmark
        (stage 04). One record per (domain, layer subset, cohort pair, method)
        with the method column carrying the method arm label and the ari and
        degen fields keyed by the k of param("clustering", "k_values").
    work("pai_matrix")        JSON, record matrix of the purity-aware arms written
        by 09_06, same record schema so that the two can be joined.
    param("gates", "g2", "baseline_method")  English fragment that identifies the
        benchmark arm of the frozen baseline in the method column. It is matched
        case-insensitively by containment, so both a bare method name and a
        decorated label match.
    param("purity", "baseline_arm")  the corresponding arm of the purity run.
    param("clustering", "k_primary"), param("clustering", "k_values")
    param("cross_implementation", "domain_a_token")  the token that marks domain A
        in a domain label; every other label is domain B.
    param("benchmark", "record_key_separator"), param("benchmark",
    "subset_separator"), param("benchmark", "pair_separator"), param("benchmark",
    "pair_separator_ascii").
    param("gates", "g2", "reproduction_tolerance"), param("gates", "g2",
    "reproduction_exact_tolerance"), param("gates", "g2", "max_listed_records").

Outputs
    None. The module prints its comparison to the console.

Usage
    python 09_08_verify_baseline_reproduction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md section 5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.config import param, work

# --- Frozen settings, read from config/params.yaml --------------------------

# Distinct ks a record was evaluated at; JSON turns the integer keys written by
# the producers into strings, so they are handled as strings here.
K_VALUES = [str(k) for k in param("clustering", "k_values")]

# The k the two baselines are compared at.
K_PRIMARY = str(param("clustering", "k_primary"))

# Benchmark arm of the frozen baseline, matched against the method column by
# containment and case-insensitively so that the fragment survives a label being
# decorated with a prefix or a suffix.
BASELINE_METHOD = param("gates", "g2", "baseline_method")

# The matching arm of the purity run.
BASELINE_ARM = param("purity", "baseline_arm")

# Token that marks domain A (the three-cohort CPTAC domain) in a domain label of
# a record; any other label is domain B.
DOMAIN_A_TOKEN = param("cross_implementation", "domain_a_token")

# The short domain keys of param("domains"). The record labels are the configured
# domain values, so the key is recovered by looking for the marker token in the
# label rather than by comparing against a literal.
DOMAIN_KEYS = list(param("domains"))
DOMAIN_A_KEY = next(
    key for key in DOMAIN_KEYS if DOMAIN_A_TOKEN in param("domains", key)
)
DOMAIN_B_KEY = next(key for key in DOMAIN_KEYS if key != DOMAIN_A_KEY)

# Label separators and the ASCII stand-in of the cohort-pair separator. All four
# are data values shared with the producers, so they are read from the
# configuration; a mismatch would silently join nothing.
RECORD_KEY_SEPARATOR = param("benchmark", "record_key_separator")
SUBSET_SEPARATOR = param("benchmark", "subset_separator")
PAIR_SEPARATOR = param("benchmark", "pair_separator")
PAIR_SEPARATOR_ASCII = param("benchmark", "pair_separator_ascii")

# Absolute ARI difference above which a shared record counts as not reproduced,
# the tighter bound within which two values count as identical, and the number of
# records a listing prints.
REPRODUCTION_TOLERANCE = float(param("gates", "g2", "reproduction_tolerance"))
REPRODUCTION_EXACT_TOLERANCE = float(
    param("gates", "g2", "reproduction_exact_tolerance")
)
MAX_LISTED_RECORDS = int(param("gates", "g2", "max_listed_records"))


def domain_key(domain_label) -> str:
    """Return the short domain key of a record's domain label.

    Args:
        domain_label: value of the domain field of a record.

    Returns:
        DOMAIN_A_KEY when the label carries the domain-A marker, DOMAIN_B_KEY
        otherwise.
    """
    return DOMAIN_A_KEY if DOMAIN_A_TOKEN in str(domain_label) else DOMAIN_B_KEY


def normalise_subset(subset_label) -> str:
    """Canonicalise a layer-subset label.

    Args:
        subset_label: value of the subset field of a record, e.g. "CNA+mRNA".

    Returns:
        The layer tokens sorted and joined again, e.g. "CNA+mRNA". The two
        producers enumerate the subsets in their own order, so the label is only
        usable as a join key once its tokens are in a canonical order.
    """
    return SUBSET_SEPARATOR.join(
        sorted(str(subset_label).replace(SUBSET_SEPARATOR, " ").split())
    )


def normalise_pair(pair_label) -> str:
    """Canonicalise a cohort-pair label to plain ASCII.

    Args:
        pair_label: value of the pair field of a record.

    Returns:
        The label with its separator replaced by the ASCII stand-in and the
        surrounding whitespace stripped, so that the join key carries no
        non-ASCII character.
    """
    return (
        str(pair_label).replace(PAIR_SEPARATOR, PAIR_SEPARATOR_ASCII).strip()
    )


def flatten_primary_k(frame: pd.DataFrame) -> pd.DataFrame:
    """Lift the ari and degen dictionaries of a record file into columns.

    Args:
        frame: record matrix as read from one of the two JSON files.

    Returns:
        The frame with the columns ari<K_PRIMARY> and deg<K_PRIMARY> added; a
        field that is not a dictionary (a record whose arm was not evaluated)
        yields None, which the later comparisons drop.
    """
    for k in K_VALUES:
        frame[f"ari{k}"] = frame["ari"].map(
            lambda value: value.get(k) if isinstance(value, dict) else None
        )
        frame[f"deg{k}"] = frame["degen"].map(
            lambda value: value.get(k) if isinstance(value, dict) else None
        )
    return frame


def add_record_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the normalised evaluation-unit key of a record file.

    Args:
        frame: record matrix carrying the fields domain, subset and pair.

    Returns:
        The frame with the columns dom (short domain key), sub (canonicalised
        layer subset) and rec (the composite joining key). The key is built from
        the three fields that identify an evaluation unit, so two records join
        exactly when they describe the same unit.
    """
    frame["dom"] = frame["domain"].map(domain_key)
    frame["sub"] = frame["subset"].map(normalise_subset)
    frame["pairn"] = frame["pair"].map(normalise_pair)
    frame["rec"] = (
        frame["dom"]
        + RECORD_KEY_SEPARATOR
        + frame["sub"]
        + RECORD_KEY_SEPARATOR
        + frame["pairn"]
    )
    return frame


def read_records(path: Path) -> pd.DataFrame:
    """Read one record matrix and prepare it for the join.

    Args:
        path: record matrix to read.

    Returns:
        The frame with its primary-k fields flattened and its evaluation-unit key
        added.
    """
    with open(path, encoding="utf-8") as handle:
        frame = pd.DataFrame(json.load(handle))
    return add_record_key(flatten_primary_k(frame))


def main() -> None:
    """Compare the two baseline arms record by record and report. Returns None."""
    value_column = f"ari{K_PRIMARY}"
    degeneracy_column = f"deg{K_PRIMARY}"

    benchmark = read_records(work("benchmark_matrix"))
    # Case-insensitive containment match, so the configured fragment identifies
    # the baseline arm without pinning its exact label. JSON has turned the
    # integer ks of the records into strings, which is why the columns are named
    # after the string ks above.
    is_baseline_method = benchmark["method"].astype(str).str.contains(
        BASELINE_METHOD, case=False, regex=False
    )
    benchmark_ari = benchmark[is_baseline_method].set_index("rec")[value_column]

    purity = read_records(work("pai_matrix"))
    # A record whose arm could not be evaluated carries ari = null and must not
    # enter the comparison.
    purity = purity[purity["ari"].notna()].copy()
    baseline = purity[purity["arm"] == BASELINE_ARM].set_index("rec")[
        [value_column, degeneracy_column]
    ]

    joint = pd.concat(
        [
            benchmark_ari.rename("bench_KMeans"),
            baseline[value_column].rename("pai_B0"),
        ],
        axis=1,
    )
    print(
        f"benchmark baseline records {len(benchmark_ari)} | "
        f"purity baseline records {len(baseline)} | "
        f"shared {joint.dropna().shape[0]}"
    )

    shared = joint.dropna()
    if len(shared):
        difference = shared["pai_B0"] - shared["bench_KMeans"]
        print(
            f"difference on the shared records: mean {difference.mean():+.6f} | "
            f"max absolute difference {difference.abs().max():.6f} | "
            f"identical {int((difference.abs() < REPRODUCTION_EXACT_TOLERANCE).sum())}"
            f"/{len(shared)}"
        )
        print(
            f"  benchmark mean {shared['bench_KMeans'].mean():.4f} | "
            f"purity baseline mean {shared['pai_B0'].mean():.4f}"
        )
        if difference.abs().max() > REPRODUCTION_TOLERANCE:
            disagreeing = difference.abs() > REPRODUCTION_TOLERANCE
            print(f"\n  records that disagree (first {MAX_LISTED_RECORDS}):")
            print(
                shared[disagreeing]
                .assign(diff=difference[disagreeing])
                .head(MAX_LISTED_RECORDS)
                .round(4)
                .to_string()
            )

    # Records held by one side only. They are the other reason a comparison of
    # the two baselines would not be like for like, so they are counted and
    # listed explicitly rather than silently dropped by the join.
    only_benchmark = sorted(set(benchmark_ari.index) - set(baseline.index))
    only_purity = sorted(set(baseline.index) - set(benchmark_ari.index))
    print(
        f"\nrecords only the benchmark holds {len(only_benchmark)} | "
        f"records only the purity run holds {len(only_purity)}"
    )
    print("only benchmark: ", only_benchmark[:MAX_LISTED_RECORDS])
    print("only purity: ", only_purity[:MAX_LISTED_RECORDS])

    # Degeneracy filtering. A degenerate clustering was already excluded from the
    # verdict by 09_07, so this section checks that both sides lose the same
    # records and that their means agree on what is left.
    print(
        f"\ndegenerate records (deg{K_PRIMARY} != 0): benchmark "
        f"{int((benchmark[degeneracy_column].fillna(0) != 0).sum())} | "
        f"purity baseline {int((baseline[degeneracy_column].fillna(0) != 0).sum())}"
    )
    benchmark_filtered = benchmark[
        is_baseline_method & (benchmark[degeneracy_column].fillna(0) == 0)
    ]
    print(
        f"benchmark baseline mean after degeneracy filtering = "
        f"{benchmark_filtered[value_column].mean():.4f} "
        f"(n={len(benchmark_filtered)})"
    )
    baseline_filtered = baseline[baseline[degeneracy_column].fillna(0) == 0]
    print(
        f"purity baseline mean after degeneracy filtering    = "
        f"{baseline_filtered[value_column].mean():.4f} "
        f"(n={len(baseline_filtered)})"
    )
    # The like-for-like comparison: both sides filtered, then restricted to the
    # units the two still share.
    filtered_joint = pd.concat(
        [
            benchmark_filtered.set_index("rec")[value_column].rename("bench"),
            baseline_filtered[value_column].rename("pai"),
        ],
        axis=1,
    ).dropna()
    print(
        f"on the shared degeneracy-filtered records: "
        f"benchmark {filtered_joint['bench'].mean():.4f} vs "
        f"purity {filtered_joint['pai'].mean():.4f} "
        f"(n={len(filtered_joint)})"
    )


if __name__ == "__main__":
    main()

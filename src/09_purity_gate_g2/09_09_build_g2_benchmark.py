"""
Derive the G2 decision threshold from the measured effect-size distribution.

Purpose
    The G2 gate asks whether a new integration algorithm beats the frozen
    baseline by a margin the evaluation design can actually resolve. That margin
    is measured here, not assumed: the module quantifies the noise floor of the
    benchmark and converts it into a minimum detectable difference (MDD), which
    is then expressed as a relative improvement over the baseline.

    Four noise sources are measured on the frozen Python benchmark matrix, and
    one of them is replicated independently on the R side:
        (i)   leave-out cohort selection: the spread of the transfer ARI of one
              (domain, layer subset, method) cell across the cohort pairs;
        (ii)  K selection: the spread of one record's ARI across the K grid;
        (iii) method comparison: the distribution of the paired ARI difference
              of two methods on the same record;
        (iv)  selection bias: the baseline is the best of a scanned set, so the
              optimistic bias of a max-of-k choice has to be added back.

    The relative threshold that follows from (iii) plus (iv) is the
    pre-registered G2 threshold. param("gates", "g2", "baseline_ari") and
    param("gates", "g2", "target_ari") hold the frozen values this module
    reproduces; the module prints the freshly measured values beside them so
    that any drift between the measurement and the pre-registration is visible.
    The threshold is therefore a property of the measured noise floor, not a
    round number chosen by convention.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")    Python benchmark records, one per (domain, layer
                                subset, cohort pair, method); the ari and degen
                                fields are keyed by the K of
                                param("clustering", "k_values").
    work("benchmark_r_matrix")  R-side benchmark records, one per (domain, layer
                                subset, cohort pair, method), used as an
                                independent replication of the cohort-selection
                                noise. The columns ari_<K> and deg_<K> mirror
                                the Python record fields.

Outputs
    work("g2_threshold")        JSON with the measured noise scales, the
                                max-of-k selection bias and the baseline levels.
                                Consumed by 09_10 and 09_11.

Usage
    python 09_09_build_g2_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md §5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from common.config import SEED, ensure_dir, param, work

# --- Configuration -----------------------------------------------------------

# All locations come from config/paths.yaml; no literal path appears below.
MATRIX_FILE = work("benchmark_matrix")
R_MATRIX_FILE = work("benchmark_r_matrix")
THRESHOLD_FILE = work("g2_threshold")

# K grid and the primary K the gate is decided at. The record fields are keyed by
# the plain string of each K, so the column names are built from the numbers.
K_VALUES = [str(k) for k in param("clustering", "k_values")]
K_PRIMARY = str(param("clustering", "k_primary"))
ARI_PRIMARY = f"ari{K_PRIMARY}"
DEGEN_PRIMARY = f"degen{K_PRIMARY}"

# Two-sided level and power of the test the threshold is derived for.
ALPHA = float(param("statistics", "alpha"))
POWER = float(param("statistics", "power"))

# Domain labels exactly as the benchmark records write them, keyed by the short
# domain key the statistics below group on.
DOMAIN_KEY_BY_LABEL = {label: key for key, label in param("domains").items()}
DOMAIN_A, DOMAIN_B = list(param("domains"))

# Separator inside a cohort-pair label; declared in params.yaml because
# downstream stages split records on it. See the note there.
PAIR_SEPARATOR = param("benchmark", "pair_separator")

# Monte-Carlo draws of the max-of-k selection-bias estimate.
NOISE_SIMULATIONS = int(param("gates", "g2", "noise_simulations"))

# Frozen G2 gate values. They are the output of this module, stored so that a
# later run can be compared against them.
PREREG_BASELINE = float(param("gates", "g2", "baseline_ari"))
PREREG_TARGET = float(param("gates", "g2", "target_ari"))
BASELINE_METHOD_FRAGMENT = str(param("gates", "g2", "baseline_method"))

# The one literal a module is permitted to carry: the global random seed. Every
# stochastic step below is seeded from it.
rng = np.random.RandomState(SEED)


# --- Data access -------------------------------------------------------------


def load_python_matrix() -> pd.DataFrame:
    """Load the Python benchmark records and unpack the per-K fields.

    Returns:
        One row per benchmark record, with the per-K transfer ARI and degeneracy
        flags expanded into columns ari<K> / degen<K>. The ari and degen fields
        of a record are dictionaries keyed by the K value, so they are unpacked
        once here and read as plain columns afterwards.
    """
    frame = pd.DataFrame(json.load(open(MATRIX_FILE)))
    for k in K_VALUES:
        frame[f"ari{k}"] = frame["ari"].map(
            lambda value: value.get(k) if isinstance(value, dict) else None
        )
    frame[DEGEN_PRIMARY] = frame["degen"].map(
        lambda value: value.get(K_PRIMARY) if isinstance(value, dict) else None
    )
    return frame


def load_r_matrix() -> pd.DataFrame:
    """Load the R-side benchmark records.

    Returns:
        The R benchmark table as read; it already carries one ari_<K> and one
        deg_<K> column per K, so no unpacking is needed.
    """
    return pd.read_csv(R_MATRIX_FILE)


def domain_key(label: str) -> str:
    """Map a domain label as written in the records to its short domain key.

    Args:
        label: value of the `domain` field, e.g. param("domains", "A").

    Returns:
        The matching key of param("domains"). An unrecognised label falls back to
        the second domain, which mirrors the else-branch of the original mapping
        and keeps a single stray label from aborting the whole derivation.
    """
    return DOMAIN_KEY_BY_LABEL.get(str(label), DOMAIN_B)


def prep(
    frame: pd.DataFrame, ari_column: str, degen_column: str | None = None
) -> pd.DataFrame:
    """Reduce a benchmark table to the analysable records.

    Args:
        frame: benchmark records.
        ari_column: name of the column holding the transfer ARI at the primary K.
        degen_column: name of the degeneracy flag at the primary K, or None to
            skip degeneracy filtering.

    Returns:
        A copy holding only records with a non-missing ARI and, when
        `degen_column` is given, only records whose clustering is not degenerate.
        The frame gains the derived fields `dom`, `sub` and `pairn`, which are
        the grouping keys of every statistic below.
    """
    reduced = frame.dropna(subset=[ari_column]).copy()
    reduced["dom"] = reduced["domain"].map(domain_key)
    # Layer subsets are compared as sets, because a record may list the layers of
    # a subset in any order; sorting the tokens before grouping makes the same
    # subset one group on both the Python and the R side.
    reduced["sub"] = reduced["subset"].map(
        lambda value: "+".join(sorted(str(value).replace("+", " ").split()))
    )
    # The cohort-pair label is reformatted to a plain ASCII separator so that the
    # two sides of a pair read the same on both platforms.
    reduced["pairn"] = (
        reduced["pair"].astype(str).str.replace(PAIR_SEPARATOR, "x").str.strip()
    )
    if degen_column and degen_column in reduced.columns:
        # A degenerate clustering (fewer than two distinct clusters) carries no
        # transfer information and would bias every spread downwards; a missing
        # flag is treated as non-degenerate.
        reduced = reduced[reduced[degen_column].fillna(0) == 0]
    return reduced


def main() -> None:
    """Derive the G2 threshold from the benchmark noise floor and write it."""
    python_records = load_python_matrix()
    r_records = load_r_matrix()
    print(f"Python records {len(python_records)} | R records {len(r_records)}")

    primary = prep(python_records, ARI_PRIMARY, DEGEN_PRIMARY)
    print(f"Python records after degeneracy removal {len(primary)}")

    # ---------- (i) noise of the leave-out cohort choice ----------
    # One cell is a (domain, layer subset, method) combination; its spread across
    # the cohort pairs is the noise a different leave-out cohort would inject.
    grouped = primary.groupby(["dom", "sub", "method"])[ARI_PRIMARY]
    sd_cohort = grouped.std(ddof=1).dropna()
    n_pair = grouped.size()
    print(f"\n(i) leave-out cohort selection ({int(n_pair.median())} cohort pairs per cell)")
    print(
        f"   within-cell SD: median {sd_cohort.median():.4f} | "
        f"mean {sd_cohort.mean():.4f} | Q90 {sd_cohort.quantile(.9):.4f}"
    )

    # Independent replication of the same statistic on the R side: if the two
    # sides disagree, the noise scale is an artefact of one implementation.
    r_primary = prep(r_records, f"ari_{K_PRIMARY}")
    grouped_r = r_primary.groupby(["dom", "sub", "method"])[f"ari_{K_PRIMARY}"]
    sd_cohort_r = grouped_r.std(ddof=1).dropna()
    print(
        f"   independent R-side replication: median {sd_cohort_r.median():.4f} | "
        f"mean {sd_cohort_r.mean():.4f}"
    )

    # ---------- (ii) noise of the K choice ----------
    # The same record evaluated at each K of the grid; the spread over K is the
    # noise a different choice of the reported K would inject.
    k_values_matrix = primary[[f"ari{k}" for k in K_VALUES]].values
    sd_k = np.nanstd(k_values_matrix, axis=1, ddof=1)
    print(f"\n(ii) K selection (K={'/'.join(K_VALUES)})")
    print(
        f"   record-level SD: median {np.nanmedian(sd_k):.4f} | "
        f"mean {np.nanmean(sd_k):.4f}"
    )

    # ---------- (iii) distribution of the paired method difference ----------
    # Every unordered pair of methods scored on the same (record) cell, which is
    # the comparison a decision between two algorithms actually makes.
    rows = []
    for (dom, sub, pair), group in primary.groupby(["dom", "sub", "pairn"]):
        scores = group.set_index("method")[ARI_PRIMARY]
        for i, method_a in enumerate(scores.index):
            for method_b in scores.index[i + 1:]:
                rows.append(
                    {
                        "dom": dom,
                        "sub": sub,
                        "pair": pair,
                        "m_a": method_a,
                        "m_b": method_b,
                        "d": scores[method_a] - scores[method_b],
                    }
                )
    differences = pd.DataFrame(rows)
    print(f"\n(iii) method paired differences ({len(differences)} pairs)")
    print(
        f"   |dARI| median {differences['d'].abs().median():.4f} | "
        f"SD {differences['d'].std(ddof=1):.4f}"
    )

    # ---------- (iv) selection bias: the baseline is a max-of-k ----------
    print("\n(iv) selection bias (the baseline is the maximum over methods, so it is optimistically biased)")
    methods_per_record = primary.groupby(["dom", "sub", "pairn"])["method"].nunique()
    k_methods = int(methods_per_record.median())
    print(f"   methods per record k = {k_methods}")
    # Monte Carlo: how far the maximum of k identically distributed methods
    # exceeds the true value, on the scale of the measured within-cell SD. This
    # is the amount by which a baseline that was *selected* as the best of the
    # set overstates the performance a genuinely independent algorithm must beat.
    simulation = []
    for _ in range(NOISE_SIMULATIONS):
        draw = rng.normal(0, sd_cohort.median(), k_methods)
        simulation.append(draw.max() - 0)
    selection_bias = float(np.mean(simulation))
    print(
        f"   max-of-{k_methods} optimistic bias ~= {selection_bias:.4f} "
        f"(on the scale of the within-cell SD {sd_cohort.median():.4f})"
    )

    # ---------- deriving the threshold ----------
    print("\n" + "=" * 84)
    print("Deriving the G2 threshold")
    print("=" * 84)

    # Evaluation-set size: the number of (domain, layer subset) cells one
    # leave-out cohort evaluation covers. This is the n of the design, and it is
    # fixed by the cohort and layer vocabularies, not chosen.
    subsets_a = primary[primary["dom"] == DOMAIN_A]["sub"].nunique()
    subsets_b = primary[primary["dom"] == DOMAIN_B]["sub"].nunique()
    print(
        f"evaluation-set size: domain A {subsets_a} layer subsets | "
        f"domain B {subsets_b} | total {subsets_a + subsets_b}"
    )

    # The baseline level the relative threshold is expressed against: the mean
    # transfer ARI of each method over all records. The best of them is the
    # reference point, so it is estimated from the same records the new
    # algorithm would be scored on.
    baseline = primary.groupby("method")[ARI_PRIMARY].mean().sort_values(ascending=False)
    print(f"\nbest baseline (record-weighted mean): {baseline.index[0]} = {baseline.iloc[0]:.4f}")
    print(
        f"runner-up: {baseline.index[1]} = {baseline.iloc[1]:.4f}  "
        f"-> observed gap only {baseline.iloc[0] - baseline.iloc[1]:.4f}"
    )
    print(
        f"range over all methods: {baseline.iloc[0]:.4f} - {baseline.iloc[-1]:.4f} "
        f"= {baseline.iloc[0] - baseline.iloc[-1]:.4f}"
    )

    # --- Cross-check against the pre-registered G2 constants ---
    # The frozen numbers are this measurement, not a convention. baseline_ari is
    # the transfer ARI of the frozen baseline arm
    # (param("gates", "g2", "baseline_method")); target_ari is the absolute ARI
    # the frozen relative gain implies. Printing the measured value beside the
    # frozen one makes any drift between the noise floor and the pre-registration
    # visible instead of silent.
    frozen_arms = [
        name
        for name in baseline.index
        if BASELINE_METHOD_FRAGMENT.lower() in name.lower()
    ]
    measured_baseline = float(baseline[frozen_arms].iloc[0]) if frozen_arms else float("nan")
    frozen_name = frozen_arms[0] if frozen_arms else "(no method matches)"
    print(
        f"\npre-registered baseline arm {frozen_name}: "
        f"measured {measured_baseline:.4f} | pre-registered {PREREG_BASELINE:.4f}"
    )
    print(f"pre-registered absolute target ARI: {PREREG_TARGET:.4f}")

    # Minimum detectable difference at the configured level and power. The
    # z-sum is written out rather than taken from a table so that a change of
    # param("statistics", "alpha") or param("statistics", "power") propagates.
    z_sum = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    for n_eval in [subsets_a, subsets_a + subsets_b, 2 * (subsets_a + subsets_b)]:
        for sd_used, tag in [
            (differences["d"].std(ddof=1), "paired-diff SD"),
            (sd_cohort.median(), "within-cohort SD"),
        ]:
            se = sd_used / np.sqrt(n_eval)
            mdd_paired = z_sum * se
            relative = mdd_paired / baseline.iloc[0]
            print(
                f"  n_eval={n_eval:3d} {tag:16s} SE={se:.4f}  MDD={mdd_paired:.4f}  "
                f"relative threshold = +{relative * 100:.1f}%"
                f"  -> absolute target {baseline.iloc[0] * (1 + relative):.4f}"
            )

    # Total threshold: the detectable difference plus the selection bias, because
    # a new algorithm must clear the baseline *as selected*, not the baseline's
    # true level.
    print("\nadding the selection bias (a new algorithm must clear the optimistic part of the max-of-k choice):")
    for n_eval in [subsets_a, subsets_a + subsets_b]:
        se = differences["d"].std(ddof=1) / np.sqrt(n_eval)
        mdd = z_sum * se
        total = mdd + selection_bias
        print(
            f"  n_eval={n_eval:3d}: MDD={mdd:.4f} + bias={selection_bias:.4f} = {total:.4f}"
            f"  -> relative threshold +{total / baseline.iloc[0] * 100:.1f}%"
            f"  absolute target {baseline.iloc[0] + total:.4f}"
        )

    output = {
        "sd_cohort_median": float(sd_cohort.median()),
        "sd_cohort_mean": float(sd_cohort.mean()),
        "sd_cohort_R_median": float(sd_cohort_r.median()),
        "sd_K_median": float(np.nanmedian(sd_k)),
        "sd_paired": float(differences["d"].std(ddof=1)),
        "n_pairs": int(len(differences)),
        "sel_bias": selection_bias,
        "k_methods": k_methods,
        "baseline_best": float(baseline.iloc[0]),
        "baseline_best_name": str(baseline.index[0]),
        "baseline_second": float(baseline.iloc[1]),
        "subs_A": int(subsets_a),
        "subs_B": int(subsets_b),
    }
    ensure_dir(THRESHOLD_FILE.parent)
    with open(THRESHOLD_FILE, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=1)
    print(f"\nwritten {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()

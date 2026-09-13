"""
Narrow the G2 noise estimate to the decision-relevant comparison and simulate
its power.

Purpose
    The noise scale used by 09_09 is averaged over every pair of methods, but a
    decision only ever compares the new algorithm with the best baseline. This
    module re-estimates the noise on exactly that comparison and reports what
    follows from it, as a separate evaluation pass from 09_09 and 09_10:

        (i)   the paired-difference SD of every challenger against the baseline,
              from which the decision-relevant noise scale is read;
        (ii)  the evaluation records needed to detect a relative improvement at
              that noise scale;
        (iii) the minimum detectable relative change at the evaluation sizes the
              design could actually reach;
        (iv)  a power simulation: the probability that an algorithm with a given
              true gain clears the frozen relative gate at the current
              evaluation size.

    The decision-relevant challenger is identified from
    param("gates", "g2", "decision_relevant_method") rather than by name, so the
    module follows the method vocabulary of the artefacts. The absolute target
    is expressed against the measured baseline, because the threshold came from
    the measured noise floor and not from a convention.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")    Python benchmark records, one per (domain, layer
                                subset, cohort pair, method); the ari and degen
                                fields are keyed by the K of
                                param("clustering", "k_values").
    work("g2_threshold")        Noise scales written by 09_09; the paired
                                difference SD is the all-pairs noise scale the
                                decision-relevant one is compared against.

Outputs
    work("g2_refined")          The decision-relevant noise scale, the paired
                                comparison table, the required sample sizes and
                                the baseline they are expressed against.

Usage
    python 09_11_build_g2_benchmark_extended.py
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
THRESHOLD_FILE = work("g2_threshold")
REFINED_FILE = work("g2_refined")

# K grid and the primary K the gate is decided at. The record fields are keyed by
# the plain string of each K, so the column names are built from the numbers.
K_VALUES = [str(k) for k in param("clustering", "k_values")]
K_PRIMARY = str(param("clustering", "k_primary"))
ARI_PRIMARY = f"ari{K_PRIMARY}"
DEGEN_PRIMARY = f"degen{K_PRIMARY}"

# Two-sided level and the power of the tests below.
ALPHA = float(param("statistics", "alpha"))
POWER = float(param("statistics", "power"))

# Domain labels exactly as the benchmark records write them, keyed by the short
# domain key the statistics below group on.
DOMAIN_KEY_BY_LABEL = {label: key for key, label in param("domains").items()}
DOMAIN_A, DOMAIN_B = list(param("domains"))

# Separator inside a cohort-pair label; declared in params.yaml because
# downstream stages split records on it. See the note there.
PAIR_SEPARATOR = param("benchmark", "pair_separator")

# Relative improvements scanned once the noise scale has been narrowed, and the
# smallest evaluation size, which is the size the design starts from.
REFINED_THRESHOLDS = list(param("gates", "g2", "refined_scenario_thresholds"))
EVAL_SIZES = list(param("gates", "g2", "evaluation_set_sizes"))
N_EVAL = int(EVAL_SIZES[0])

# True relative improvements simulated for the power curve, and the number of
# draws per point (a Monte-Carlo figure rather than an exact power).
POWER_GAINS = list(param("gates", "g2", "power_simulation_gains"))
POWER_SIMULATIONS = int(param("gates", "g2", "power_simulations"))

# Frozen constants of the gate: the relative gain the threshold was derived for
# and the fragment naming the decision-relevant challenger.
PLACEHOLDER_GAIN = float(param("gates", "g2", "placeholder_relative_gain"))
DECISION_RELEVANT_METHOD = str(param("gates", "g2", "decision_relevant_method"))

# Cap on how many challengers receive a paired test: a reporting limit that keeps
# the table readable. The decision-relevant challenger always ranks within it.
CANDIDATE_LIMIT = 6

# z_{1-alpha/2} + z_{power} at the configured level and power, rounded to the one
# decimal (2.8) that the frozen sample-size and minimum-detectable-change figures
# were computed with, so that they reproduce exactly.
Z_SUM = round(stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER), 1)

# The one literal a module is permitted to carry: the global random seed. The
# power simulation below draws from it.
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


def domain_key(label: str) -> str:
    """Map a domain label as written in the records to its short domain key.

    Args:
        label: value of the `domain` field, e.g. param("domains", "A").

    Returns:
        The matching key of param("domains"). An unrecognised label falls back to
        the second domain, which mirrors the else-branch of the original mapping.
    """
    return DOMAIN_KEY_BY_LABEL.get(str(label), DOMAIN_B)


def prep(frame: pd.DataFrame, ari_column: str, degen_column: str | None = None) -> pd.DataFrame:
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
    # subset one group.
    reduced["sub"] = reduced["subset"].map(
        lambda value: "+".join(sorted(str(value).replace("+", " ").split()))
    )
    # The cohort-pair label is reformatted to a plain ASCII separator so that the
    # pair identity is read the same way everywhere.
    reduced["pairn"] = (
        reduced["pair"].astype(str).str.replace(PAIR_SEPARATOR, "x").str.strip()
    )
    if degen_column and degen_column in reduced.columns:
        # A degenerate clustering (fewer than two distinct clusters) carries no
        # transfer information; a missing flag is treated as non-degenerate.
        reduced = reduced[reduced[degen_column].fillna(0) == 0]
    return reduced


def main() -> None:
    """Narrow the noise scale to the decision-relevant comparison and write it."""
    with open(THRESHOLD_FILE, encoding="utf-8") as handle:
        threshold = json.load(handle)

    records = prep(load_python_matrix(), ARI_PRIMARY, DEGEN_PRIMARY)
    # One row per (domain, layer subset, cohort pair) cell, one column per
    # method: the wide table the paired comparison is read from.
    records["rec"] = records["dom"] + "|" + records["sub"] + "|" + records["pairn"]
    wide = records.pivot_table(index="rec", columns="method", values=ARI_PRIMARY)
    print(f"wide table: {wide.shape[0]} records x {wide.shape[1]} methods")
    baseline = wide.mean().sort_values(ascending=False)
    best = baseline.index[0]
    best_value = baseline.iloc[0]
    print(f"baseline method {best} = {best_value:.4f}\n")

    # ---------- (i) paired difference against the baseline ----------
    print("=" * 92)
    print("(i) paired-difference SD of every method against the baseline (the comparison the decision actually involves)")
    print("=" * 92)
    candidates = [method for method in baseline.index if method != best][:CANDIDATE_LIMIT]
    comparisons = []
    for method in candidates:
        # The same cell scored by both methods; only cells present in both enter,
        # which is what makes this a paired comparison.
        difference = (wide[method] - wide[best]).dropna()
        sd = difference.std(ddof=1)
        t, p = stats.ttest_rel(wide[method].dropna()[difference.index], wide[best].reindex(difference.index))
        comparisons.append(
            {
                "vs": method,
                "mean_diff": round(difference.mean(), 4),
                "sd_diff": round(sd, 4),
                "n": len(difference),
                "paired_p": float(p),
            }
        )
        print(
            f"  {method:30s} mean diff {difference.mean():+.4f}  SD {sd:.4f}  "
            f"n={len(difference)}  paired p={p:.3g}"
        )
    sd_top = pd.DataFrame(comparisons).sort_values("sd_diff")
    print(
        f"\n  paired SD against the best baseline ranges over "
        f"{sd_top['sd_diff'].min():.4f} - {sd_top['sd_diff'].max():.4f}"
    )
    # The decision-relevant challenger is the co-association consensus. It is
    # located by the configured fragment rather than by a literal name, so the
    # module follows the method vocabulary of the artefacts. Exactly one match is
    # required: an ambiguous fragment would silently pick the wrong noise scale.
    matches = [
        row["vs"]
        for row in comparisons
        if DECISION_RELEVANT_METHOD.lower() in str(row["vs"]).lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one challenger matching "
            f"{DECISION_RELEVANT_METHOD!r}, found {matches!r}"
        )
    sd_relevant = float(
        pd.DataFrame(comparisons).set_index("vs").loc[matches[0], "sd_diff"]
    )
    print(
        f"  closest challenger ({matches[0]}) SD = {sd_relevant:.4f}  "
        f"<- decision-relevant noise scale"
    )

    # ---------- (ii) sample size at the decision-relevant scale ----------
    print("\n" + "=" * 92)
    print("(ii) sample size required to clear the threshold (decision-relevant noise)")
    print("=" * 92)
    print(
        f"  {'rel. threshold':>14s} {'abs. delta':>11s} "
        f"{'n (relevant)':>13s} {'n (all pairs)':>14s}"
    )
    with open(THRESHOLD_FILE, encoding="utf-8") as handle:
        sd_all = json.load(handle)["sd_paired"]
    required = []
    for relative in REFINED_THRESHOLDS:
        delta = relative * best_value
        n_relevant = (Z_SUM * sd_relevant / delta) ** 2
        n_allpairs = (Z_SUM * sd_all / delta) ** 2
        required.append(
            {
                "rel": relative,
                "abs": round(delta, 4),
                "n_relevant": int(np.ceil(n_relevant)),
                "n_allpairs": int(np.ceil(n_allpairs)),
            }
        )
        print(
            f"  {('+' + str(int(relative * 100)) + '%'):>14s} {delta:11.4f} "
            f"{int(np.ceil(n_relevant)):13d} {int(np.ceil(n_allpairs)):14d}"
        )

    # ---------- (iii) minimum detectable change at reachable sizes ----------
    print("\n" + "=" * 92)
    print("(iii) minimum detectable relative change at the evaluation sizes the design could reach (MDD)")
    print("=" * 92)
    for n in EVAL_SIZES:
        for sd, tag in [(sd_relevant, "challenger SD"), (sd_all, "all-pairs SD")]:
            mdd = Z_SUM * sd / np.sqrt(n)
            print(
                f"  n={n:3d} {tag:16s} MDD={mdd:.4f} -> relative threshold "
                f"+{mdd / best_value * 100:.1f}%  (absolute target {best_value + mdd:.4f})"
            )

    # ---------- (iv) power simulation ----------
    print("\n" + "=" * 92)
    print(
        f"(iv) power simulation: probability that an algorithm with a true gain clears "
        f"the +{PLACEHOLDER_GAIN * 100:.0f}% gate at n={N_EVAL}"
    )
    print("=" * 92)
    print("  (assuming the paired-difference SD of the new algorithm against the baseline equals the closest challenger's SD)")
    for true_relative in POWER_GAINS:
        true_delta = true_relative * best_value
        hits = 0
        for _ in range(POWER_SIMULATIONS):
            draw = rng.normal(true_delta, sd_relevant, N_EVAL)
            # One-sided paired t-test at the configured alpha: a simulated gain
            # is called significant when its t statistic exceeds the critical
            # value with N_EVAL - 1 degrees of freedom.
            tstat = draw.mean() / (draw.std(ddof=1) / np.sqrt(N_EVAL))
            if tstat > stats.t.ppf(1 - ALPHA, N_EVAL - 1):
                hits += 1
        print(
            f"  true gain +{int(true_relative * 100):3d}%  ->  probability of being "
            f"called significant at n={N_EVAL} = {hits / POWER_SIMULATIONS * 100:5.1f}%"
        )

    output = {
        "baseline": best,
        "baseline_value": float(best_value),
        "sd_relevant": sd_relevant,
        "sd_allpairs": float(sd_all),
        "candidate_pairs": comparisons,
        "required_n": required,
    }
    ensure_dir(REFINED_FILE.parent)
    with open(REFINED_FILE, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=1)
    print(f"\nwritten {REFINED_FILE}")


if __name__ == "__main__":
    main()

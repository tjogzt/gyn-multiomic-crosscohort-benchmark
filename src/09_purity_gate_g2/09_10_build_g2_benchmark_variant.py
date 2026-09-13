"""
Run the decisive checks on the G2 threshold: winner flips, required sample size
and replicability.

Purpose
    The threshold derived by 09_09 is only usable if the benchmark ranking it is
    compared against is itself stable. This module tests that directly, and it is
    deliberately a separate evaluation pass from 09_09 and 09_11:

        (i)   winner flip: on one (domain, layer subset) cell, does the winning
              method change with the leave-out cohort? A ranking that flips is
              not a ranking the gate can lean on, whatever its mean.
        (ii)  required sample size: how many evaluation records are needed for a
              given relative improvement to become detectable at the configured
              level and power?
        (iii) feasibility of the alternative G2 design: a replicability
              criterion that asks a method to rank first in a fixed fraction of
              the leave-out cohorts of a cell, rather than in mean ARI.

    Nothing here re-derives the threshold; it consumes
    work("g2_threshold") and decides whether the threshold can be used. The
    pre-registered relative gains and the criterion constants all come from
    param("gates", "g2", ...).

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
                                difference SD and the within-cohort SD set the
                                scale of every sample-size figure below.

Outputs
    work("g2_champ_flip")       Per (domain, layer subset) cell: the number of
                                leave-out cohort pairs, the number of distinct
                                winners and the winner list.
    work("g2_required_n")       Required evaluation records per relative
                                threshold.
    work("g2_replicability")    Per method: the number of cells in which it ranks
                                first in the required fraction of cohort pairs.

Usage
    python 09_10_build_g2_benchmark_variant.py
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
FLIP_FILE = work("g2_champ_flip")
REQUIRED_N_FILE = work("g2_required_n")
REPLICABILITY_FILE = work("g2_replicability")
FINAL_FILE = work("g2_final")

# K grid and the primary K the gate is decided at. The record fields are keyed by
# the plain string of each K, so the column names are built from the numbers.
K_VALUES = [str(k) for k in param("clustering", "k_values")]
K_PRIMARY = str(param("clustering", "k_primary"))
ARI_PRIMARY = f"ari{K_PRIMARY}"
DEGEN_PRIMARY = f"degen{K_PRIMARY}"

# Two-sided level and power of the test the sample-size table is computed for.
ALPHA = float(param("statistics", "alpha"))
POWER = float(param("statistics", "power"))

# Domain labels exactly as the benchmark records write them, keyed by the short
# domain key the statistics below group on.
DOMAIN_KEY_BY_LABEL = {label: key for key, label in param("domains").items()}
DOMAIN_A, DOMAIN_B = list(param("domains"))

# Separator inside a cohort-pair label; declared in params.yaml because
# downstream stages split records on it. See the note there.
PAIR_SEPARATOR = param("benchmark", "pair_separator")

# Relative improvements scanned by the required-sample-size table, and the
# original placeholder gain that the frozen threshold superseded.
SCENARIO_THRESHOLDS = list(param("gates", "g2", "scenario_relative_thresholds"))
PLACEHOLDER_GAIN = float(param("gates", "g2", "placeholder_relative_gain"))

# Constants of the alternative (replicability) gate design: a unit only enters
# the count when it has at least `MIN_PAIRS_PER_UNIT` leave-out cohorts, and a
# method counts as replicable in a unit when it ranks first in at least
# `WIN_FRACTION` of them.
MIN_PAIRS_PER_UNIT = int(param("gates", "g2", "min_pairs_per_unit"))
WIN_FRACTION = float(param("gates", "g2", "replicability_win_fraction"))

# z_{1-alpha/2} + z_{power} at the configured level and power, rounded to the one
# decimal (2.8) that the frozen required-sample-size tables were computed with,
# so that those tables reproduce exactly.
Z_SUM = round(stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER), 1)

# The one literal a module is permitted to carry: the global random seed. The
# decisive checks below are deterministic and do not draw from this generator,
# which is instantiated only so that every module of the stage shares one seed.
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
        the second domain, which mirrors the else-branch of the original mapping
        and keeps a single stray label from aborting the whole check.
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
    """Run the decisive checks on the G2 threshold and write their tables."""
    with open(THRESHOLD_FILE, encoding="utf-8") as handle:
        threshold = json.load(handle)

    records = prep(load_python_matrix(), ARI_PRIMARY, DEGEN_PRIMARY)
    # Evaluation-set size per domain: the (domain, layer subset) cells the fixed
    # leave-out evaluation covers, counted exactly as in 09_09.
    subsets_a = records[records["dom"] == DOMAIN_A]["sub"].nunique()
    subsets_b = records[records["dom"] == DOMAIN_B]["sub"].nunique()
    print(
        f"analysis records {len(records)} | layer subsets in domain A / B "
        f"{subsets_a}/{subsets_b}"
    )

    # The ranking the gate is compared against: mean transfer ARI per method.
    baseline = records.groupby("method")[ARI_PRIMARY].mean().sort_values(ascending=False)
    best = baseline.index[0]
    best_value = baseline.iloc[0]
    print(f"\nbest baseline: {best} = {best_value:.4f}")

    # ---------- (i) winner flip ----------
    print("\n" + "=" * 84)
    print("(i) does the winner flip with the leave-out cohort")
    print("=" * 84)
    # Winner of every (domain, layer subset, cohort pair) cell, then one row per
    # cell summarising whether that winner was the same across its pairs.
    champions = {}
    for (dom, sub, pair), group in records.groupby(["dom", "sub", "pairn"]):
        champions[(dom, sub, pair)] = group.loc[group[ARI_PRIMARY].idxmax(), "method"]
    champion_series = pd.Series(champions)
    flips = []
    for (dom, sub), group in champion_series.groupby(level=[0, 1]):
        winners = list(group.values)
        flips.append(
            {
                "dom": dom,
                "sub": sub,
                "n_pairs": len(winners),
                "n_unique_winner": len(set(winners)),
                "consistent": len(set(winners)) == 1,
                "winners": " | ".join(winners),
            }
        )
    flip_frame = pd.DataFrame(flips)
    print(
        f"{len(flip_frame)} (domain x layer subset) cells, each with "
        f"{int(flip_frame['n_pairs'].median())} leave-out cohort pairs"
    )
    print(
        f"  cells with a fully consistent winner: {int(flip_frame['consistent'].sum())}"
        f"/{len(flip_frame)} = {flip_frame['consistent'].mean() * 100:.1f}%"
    )
    print(
        f"  cells with a flipped winner: {int((~flip_frame['consistent']).sum())}"
        f"/{len(flip_frame)} = {(~flip_frame['consistent']).mean() * 100:.1f}%"
    )
    print(f"  mean number of distinct winners per cell: {flip_frame['n_unique_winner'].mean():.2f}")
    print(
        "\n  times each method was selected as the winner "
        "(over %d cell x cohort-pair units):" % int(flip_frame["n_pairs"].sum())
    )
    print("  " + champion_series.value_counts().to_string().replace("\n", "\n  "))
    ensure_dir(FLIP_FILE.parent)
    flip_frame.to_csv(FLIP_FILE, index=False)

    # ---------- (ii) required sample size ----------
    print("\n" + "=" * 84)
    print("(ii) records required to detect a given relative threshold")
    print("=" * 84)
    sd_paired = threshold["sd_paired"]
    sd_cohort = threshold["sd_cohort_median"]
    print(
        f"  noise scale: paired-difference SD = {sd_paired:.4f} | "
        f"within-cohort SD = {sd_cohort:.4f}"
    )
    print(
        f"  {'target rel.':>14s} {'abs. delta':>11s} "
        f"{'n (paired diff)':>16s} {'n (within cohort)':>18s}"
    )
    required = []
    for relative in SCENARIO_THRESHOLDS:
        delta = relative * best_value
        # Two-sided test of a mean difference at the configured level and power;
        # the same z-sum scales both noise models.
        n_paired = (Z_SUM * sd_paired / delta) ** 2
        n_cohort = (Z_SUM * sd_cohort / delta) ** 2
        required.append(
            {
                "rel_threshold": relative,
                "abs_delta": round(delta, 4),
                "n_needed_paired": int(np.ceil(n_paired)),
                "n_needed_cohort": int(np.ceil(n_cohort)),
            }
        )
        print(
            f"  {('+' + str(int(relative * 100)) + '%'):>14s} {delta:11.4f} "
            f"{int(np.ceil(n_paired)):16d} {int(np.ceil(n_cohort)):18d}"
        )
    print(
        f"\n  current evaluation-set size: domain A {subsets_a} + domain B {subsets_b} "
        f"= {subsets_a + subsets_b} (domain x layer subset) cells"
    )
    # The prompt that forced the threshold to be re-derived: the original
    # placeholder gain is far below what this design can resolve.
    print(
        f"  -> the +{PLACEHOLDER_GAIN * 100:.0f}% placeholder is NOT detectable at the "
        f"current size (needs "
        f"{int(np.ceil((Z_SUM * sd_cohort / (PLACEHOLDER_GAIN * best_value)) ** 2))}-"
        f"{int(np.ceil((Z_SUM * sd_paired / (PLACEHOLDER_GAIN * best_value)) ** 2))} records)"
    )
    pd.DataFrame(required).to_csv(REQUIRED_N_FILE, index=False)

    # ---------- (iii) feasibility of the replicability criterion ----------
    print("\n" + "=" * 84)
    print("(iii) alternative G2 design: feasibility of the replicability criterion")
    print("=" * 84)
    print(
        f"  criterion: on the FIXED leave-out cohorts, a method ranks first in at "
        f"least {WIN_FRACTION:.0%} of the cohort pairs of a cell (not a "
        f"single-cohort percentage)"
    )
    print("  basis: the measured winner-flip rates are")
    top2 = baseline.head(2).index.tolist()
    print(
        f"  current top 2: {top2[0]} ({baseline.iloc[0]:.4f}) / "
        f"{top2[1]} ({baseline.iloc[1]:.4f}), gap {baseline.iloc[0] - baseline.iloc[1]:.4f}"
    )
    # For every method: in how many (domain, subset) cells does it rank first in
    # at least `WIN_FRACTION` of that cell's cohort pairs. The winner is read from
    # the full record set of the cell, not only from the method's own rows, so the
    # comparison is against every other method present on that cell.
    rows = []
    for method in baseline.index:
        method_frame = records[records["method"] == method]
        ok = 0
        total = 0
        for (dom, sub), group in method_frame.groupby(["dom", "sub"]):
            wins = 0
            n = 0
            for pair, _ in group.groupby("pairn"):
                all_methods = records[
                    (records["dom"] == dom)
                    & (records["sub"] == sub)
                    & (records["pairn"] == pair)
                ]
                if len(all_methods) == 0:
                    continue
                n += 1
                if all_methods.loc[all_methods[ARI_PRIMARY].idxmax(), "method"] == method:
                    wins += 1
            # A cell with fewer than the required number of leave-out cohorts
            # cannot express a fraction, so it does not enter the denominator.
            if n >= MIN_PAIRS_PER_UNIT:
                total += 1
                if wins / n >= WIN_FRACTION:
                    ok += 1
        rows.append(
            {
                "method": method,
                "mean_ari": round(baseline[method], 4),
                "units_2of3_win": ok,
                "units_total": total,
                "frac": round(ok / total, 4) if total else None,
            }
        )
    replicability = pd.DataFrame(rows).sort_values("mean_ari", ascending=False)
    print("\n  " + replicability.to_string(index=False).replace("\n", "\n  "))
    ensure_dir(REPLICABILITY_FILE.parent)
    replicability.to_csv(REPLICABILITY_FILE, index=False)
    print(
        f"\n  -> the highest '>= {WIN_FRACTION:.0%} replicable win rate' of the current "
        f"baseline is only {replicability['frac'].max() * 100:.1f}% "
        f"({replicability.loc[replicability['frac'].idxmax(), 'method']}), i.e. no "
        f"existing method meets the criterion"
    )

    output = {
        "champ_flip_rate": float((~flip_frame["consistent"]).mean()),
        "n_units": len(flip_frame),
        "mean_unique_winners": float(flip_frame["n_unique_winner"].mean()),
        "required_n": required,
        "replicability": replicability.to_dict(orient="records"),
        "best_method": best,
        "best_value": float(best_value),
    }
    with open(FINAL_FILE, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=1)
    print(f"\nwritten {FINAL_FILE}")


if __name__ == "__main__":
    main()

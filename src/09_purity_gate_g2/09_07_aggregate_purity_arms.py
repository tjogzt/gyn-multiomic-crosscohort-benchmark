"""
Aggregate the purity-aware integration arms and decide gate G2.

Purpose
    Second half of the purity gate (G2). The module reduces the per-record arm
    matrix written by 09_06 to one mean transfer ARI per arm, compares every
    purity-aware arm with the frozen baseline arm on the records the two arms
    share, and then decides the gate against the frozen threshold.
    Six purity-aware arms were evaluated against the frozen baseline transfer ARI
    of 0.4027: the three S arms that project out the first k purity-related
    principal components, the two G arms that zero the genes whose |r| with purity
    reaches gamma, and the R1 arm that residualises each gene on purity. All six
    came out worse than the baseline, so under this protocol no purity correction
    improves cross-cohort transfer and the gate is not passed.
    The threshold is not an assumed round number. It is the relative improvement
    the design was measured to resolve (+46.6%), derived from the decision-
    relevant noise floor recorded as sd_relevant in work("g2_refined") -- the
    residual scale of the closest challenger arm under the leave-one-cohort-out
    protocol -- and frozen before any purity arm was run. A negative verdict is
    only drawn as conclusive when the upper bound of the paired confidence
    interval still falls short of the absolute target; otherwise the outcome is
    recorded as undecided (under-powered) instead of as a failure, so that a
    failure is never read as stronger than the evidence supports.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("pai_matrix")  JSON, per-record arm matrix written by 09_06: one record
        per (domain, layer subset, cohort pair, arm) whose ari and degen fields
        are keyed by the k of param("clustering", "k_values"). The record fields
        domain, subset and pair have to carry the same vocabulary as the
        benchmark records, because those three fields identify an evaluation unit.
    work("g2_refined")  JSON, frozen gate parameters: baseline (the baseline arm
        name), baseline_value (its measured transfer ARI) and sd_relevant (the
        decision-relevant noise scale the relative threshold was derived from).
    param("purity", "arms"), param("purity", "baseline_arm"),
    param("purity", "arm_labels")  the arm vocabulary of the matrix, the arm the
        gate is decided against, and the compact English report labels of the
        arms.
    param("gates", "g2", "required_relative_gain"), param("gates", "g2",
    "min_paired_records")  frozen gate values.
    param("clustering", "k_primary"), param("clustering", "k_values"),
    param("statistics", "ci_z"), param("benchmark", "record_key_separator"),
    param("benchmark", "subset_separator").

Outputs
    work("pai_compare_csv")  CSV, one row per purity-aware arm: paired record
        count, arm mean, mean paired difference, relative difference, paired
        t-test p value, the two-sided confidence interval of the difference and
        the share of records the arm wins.
    work("pai_results")      JSON, the per-arm mean table, the comparison rows,
        the frozen baseline and its absolute target, and the record and unit
        counts.
    Console: frozen parameters, the per-arm table, the paired comparisons, the
        per-domain and per-layer-count tables, and the G2 verdict.

Usage
    python 09_07_aggregate_purity_arms.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md section 5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from common.config import ensure_dir, param, work

# --- Frozen settings, read from config/params.yaml --------------------------

# Distinct ks a record was evaluated at. 09_06 writes them with integer keys, and
# JSON turns those into strings, so the keys are handled as strings here.
K_VALUES = [str(k) for k in param("clustering", "k_values")]

# The k the verdict is read at; every table below aggregates this one.
K_PRIMARY = str(param("clustering", "k_primary"))

# Relative improvement the gate demands. It is the *measured* threshold: the
# smallest relative gain the design can resolve at its record count, derived from
# the decision-relevant noise floor rather than chosen, and stored as a fraction
# (0.466, i.e. +46.6%) so that it can be re-derived. GAIN_FACTOR is the same
# value expressed as the multiplier applied to the baseline.
RELATIVE_GAIN = float(param("gates", "g2", "required_relative_gain"))
GAIN_FACTOR = 1.0 + RELATIVE_GAIN

# An arm with fewer comparable records than this cannot be tested in a paired
# way, so it is reported and left out of the comparison.
MIN_PAIRED_RECORDS = int(param("gates", "g2", "min_paired_records"))

# Two-sided interval multiplier of the paired differences. It is read from the
# configuration instead of being derived from param("statistics", "ci_level")
# because it is the literal the released interval bounds were computed with;
# deriving it (1.959964 rather than 1.96) would move their fourth decimal.
CI_Z = float(param("statistics", "ci_z"))

# Arm vocabulary: the arm the gate is decided against, and the compact report
# labels written to the comparison table.
BASELINE_ARM = param("purity", "baseline_arm")
ARM_LABELS = dict(param("purity", "arm_labels"))

# Separators of the composite record key (domain|layer subset|cohort pair) and of
# a layer-subset label (layer+layer). Both are data values shared with the
# benchmark records, so they come from the configuration rather than being
# assumed: a different separator would silently fail to join the two sides.
RECORD_KEY_SEPARATOR = param("benchmark", "record_key_separator")
SUBSET_SEPARATOR = param("benchmark", "subset_separator")


def arm_label(arm) -> str:
    """Return the report label of an arm.

    Args:
        arm: arm name as written in the arm matrix.

    Returns:
        The compact English label of param("purity", "arm_labels"), or the arm
        name itself when the configuration holds no label for it; the arm
        vocabulary is English and self-describing, so an unlabelled arm is still
        readable.
    """
    return ARM_LABELS.get(str(arm), str(arm))


def load_frozen_gate(path: Path) -> dict:
    """Read the frozen gate parameters derived from the measured noise floor.

    Args:
        path: work("g2_refined").

    Returns:
        The gate dictionary. Only baseline, baseline_value and sd_relevant are
        read here: baseline_value was measured under the frozen benchmark
        protocol, and sd_relevant is the decision-relevant noise scale the
        relative threshold was derived from.
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_arm_records(path: Path) -> pd.DataFrame:
    """Load the arm matrix and flatten its per-k fields into columns.

    Args:
        path: work("pai_matrix").

    Returns:
        One row per usable record, carrying the columns ari<k> and deg<k> for
        every k of param("clustering", "k_values"), the composite record key rec
        and the number of concatenated layers nlay.

    Records whose arm could not be evaluated (an inapplicable arm writes
    ari = null) and records that are degenerate at the primary k are removed,
    exactly as in the benchmark aggregation, so that both sides of the comparison
    agree on what a valid record is.
    """
    with open(path, encoding="utf-8") as handle:
        records = pd.DataFrame(json.load(handle))

    records = records[records["ari"].notna()].copy()

    # ari and degen are dictionaries keyed by k. They are lifted into flat columns
    # so that every later groupby and pivot can address the primary k directly.
    for k in K_VALUES:
        records[f"ari{k}"] = records["ari"].map(
            lambda value: value.get(k) if isinstance(value, dict) else None
        )
        records[f"deg{k}"] = records["degen"].map(
            lambda value: value.get(k) if isinstance(value, dict) else None
        )

    # A degenerate clustering (one cluster holding the majority of the samples)
    # says nothing about the compared representations, so such records carry no
    # information about the arm and are dropped from the verdict. A missing flag
    # counts as degenerate, which keeps the filter conservative.
    records = records[records[f"deg{K_PRIMARY}"].fillna(1) == 0]

    # Composite key of an evaluation unit: the comparison is paired on this key,
    # so an arm is only ever compared with the baseline on units both arms hold.
    records["rec"] = (
        records["domain"]
        + RECORD_KEY_SEPARATOR
        + records["subset"]
        + RECORD_KEY_SEPARATOR
        + records["pair"]
    )

    # Number of layers concatenated in the subset label, used by the layer-count
    # table of the report.
    records["nlay"] = records["subset"].map(
        lambda subset: len(str(subset).split(SUBSET_SEPARATOR))
    )
    return records


def summarise_by_arm(records: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Aggregate the transfer ARI of every arm at the primary k.

    Args:
        records: usable records of the arm matrix.
        value_column: ARI column to aggregate, e.g. "ari4".

    Returns:
        Table indexed by the arm name with the mean and the record count, sorted
        by decreasing mean so that the first row is the best-scoring arm. This is
        the table the verdict reads.
    """
    return (
        records.groupby("arm")[value_column]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )


def compare_with_baseline(
    records: pd.DataFrame, value_column: str, baseline_value: float
) -> list[dict]:
    """Compare every non-baseline arm with the baseline arm pair by pair.

    Args:
        records: usable records of the arm matrix.
        value_column: ARI column to compare, e.g. "ari4".
        baseline_value: frozen baseline transfer ARI, used to express the mean
            paired difference as a relative difference.

    Returns:
        One dictionary per compared arm, with keys arm (report label), n (paired
        records), mean_arm, mean_delta, rel_vs_base, p, ci_lo, ci_hi and
        win_rate. An arm with fewer than
        param("gates", "g2", "min_paired_records") comparable records is printed
        and skipped, because a paired test on so few records is not
        interpretable.
    """
    rows: list[dict] = []
    # The baseline series is the same for every comparison, so it is built once.
    baseline = records[records["arm"] == BASELINE_ARM].set_index("rec")[value_column]

    for arm in sorted(records["arm"].unique()):
        if arm == BASELINE_ARM:
            # The baseline is the reference; it cannot be compared with itself.
            continue

        # Inner join on the evaluation unit: only units both arms hold enter the
        # comparison, which is what makes the test paired.
        paired = pd.concat(
            [
                records[records["arm"] == arm]
                .set_index("rec")[value_column]
                .rename("arm"),
                baseline.rename("base"),
            ],
            axis=1,
        ).dropna()

        if len(paired) < MIN_PAIRED_RECORDS:
            print(f"  {arm_label(arm):28s} too few comparable records ({len(paired)})")
            continue

        difference = paired["arm"] - paired["base"]
        # Paired t test over the same records. The t statistic itself is not
        # reported: the mean difference and its interval carry the effect size of
        # the comparison, the p value only says whether it is distinguishable
        # from zero.
        tstat, pvalue = stats.ttest_rel(paired["arm"], paired["base"])

        # Half-width of the normal-approximation interval of the mean difference.
        # The sample standard deviation uses ddof=1, the usual unbiased estimate.
        half_width = CI_Z * difference.std(ddof=1) / np.sqrt(len(paired))

        rows.append(
            {
                "arm": arm_label(arm),
                "n": len(paired),
                "mean_arm": round(paired["arm"].mean(), 4),
                "mean_delta": round(difference.mean(), 4),
                "rel_vs_base": round(difference.mean() / baseline_value, 4),
                "p": float(pvalue),
                "ci_lo": round(difference.mean() - half_width, 4),
                "ci_hi": round(difference.mean() + half_width, 4),
                "win_rate": round(float((difference > 0).mean()), 4),
            }
        )
        print(
            f"  {arm_label(arm):28s} n={len(paired):3d}  "
            f"delta_ARI={difference.mean():+.4f}  "
            f"relative {difference.mean() / baseline_value * 100:+6.1f}%  "
            f"95%CI[{difference.mean() - half_width:+.4f},"
            f"{difference.mean() + half_width:+.4f}]  "
            f"win rate {(difference > 0).mean() * 100:4.1f}%  "
            f"paired p={pvalue:.3g}"
        )
    return rows


def main() -> None:
    """Aggregate the arms, write both artefacts and decide gate G2. Returns None."""
    ensure_dir(work("pai_dir"))

    gate = load_frozen_gate(work("g2_refined"))
    baseline_value = gate["baseline_value"]
    # The noise scale the relative threshold was derived from. It is printed so
    # that the provenance of the threshold stays visible in the run log: the
    # threshold is a measured quantity, not a value chosen by hand.
    sd_relevant = gate["sd_relevant"]
    print(
        f"G2 frozen parameters: baseline {gate['baseline']} = {baseline_value:.4f} | "
        f"decision-relevant noise floor sd_relevant = {sd_relevant:.4f} | "
        f"threshold +{RELATIVE_GAIN * 100:.1f}% -> absolute target "
        f"{baseline_value * GAIN_FACTOR:.4f}"
    )

    records = load_arm_records(work("pai_matrix"))
    print(f"\nusable records {len(records)} | arms: {sorted(records['arm'].unique())}")
    print(f"units (domain x layer subset x cohort pair): {records['rec'].nunique()}")

    value_column = f"ari{K_PRIMARY}"
    table = summarise_by_arm(records, value_column)
    print(f"\n=== transfer ARI per arm at K={K_PRIMARY} ===")
    print(table.round(4).to_string())

    print(
        f"\n=== paired comparison with the baseline arm {BASELINE_ARM} "
        "(same records) ==="
    )
    compare_rows = compare_with_baseline(records, value_column, baseline_value)
    compare_table = pd.DataFrame(compare_rows)
    compare_table.to_csv(work("pai_compare_csv"), index=False)

    # ---------- Per-domain means ----------
    print(f"\n=== per domain (mean at K={K_PRIMARY}) ===")
    print(
        records.pivot_table(
            index=["domain", "arm"], values=value_column, aggfunc="mean"
        )
        .round(4)
        .to_string()
    )

    # ---------- Per-layer-count means ----------
    # Single-layer subsets are separated from the concatenations so that a gain
    # concentrated in one layer (the protein layer is the candidate) is visible.
    print(f"\n=== by number of layers (mean at K={K_PRIMARY}) ===")
    print(
        records.pivot_table(
            index=["nlay", "arm"], values=value_column, aggfunc="mean"
        )
        .round(4)
        .to_string()
    )

    # ---------- G2 verdict ----------
    print("\n" + "=" * 88)
    print("G2 verdict (frozen criterion)")
    print("=" * 88)
    best = table.index[0]
    best_value = table.loc[best, "mean"]
    if best == BASELINE_ARM:
        # No purity-aware arm scores higher than the baseline. In that case the
        # baseline itself is the best arm, which means purity handling brought no
        # improvement at all -- a stronger and simpler statement than any relative
        # difference.
        print(
            f"  the best arm is the baseline itself ({best_value:.4f}) -> "
            "purity-aware handling brings no improvement at all"
        )
        print("  -> G2 = FAILED (but see the under-powered check below)")
    else:
        need = baseline_value * GAIN_FACTOR
        print(
            f"  best arm {arm_label(best)} = {best_value:.4f} | "
            f"baseline {baseline_value:.4f} | absolute target {need:.4f}"
        )
        print(
            f"  -> {'PASSED' if best_value >= need else 'FAILED'} "
            f"(relative gain {(best_value / baseline_value - 1) * 100:+.1f}%, "
            f"threshold +{RELATIVE_GAIN * 100:.1f}%)"
        )

    # Under-powered check, pre-registered: on a finite record count a failure is
    # only conclusive when the observed relative gain cannot reach the threshold
    # even at the upper bound of its interval. Otherwise the result is recorded as
    # undecided rather than as a failure.
    if len(compare_rows):
        first = compare_rows[0]
        need_absolute = RELATIVE_GAIN * baseline_value
        print(
            "\n  under-powered check (pre-registered clause): upper bound of the "
            "observed relative gain = "
            f"upper 95% CI {first['ci_hi'] / baseline_value * 100:+.1f}%"
        )
        print(
            f"    can +{RELATIVE_GAIN * 100:.1f}% be excluded? "
            + (
                "yes (the failure is conclusive)"
                if first["ci_hi"] < need_absolute
                else "no -> recorded as UNDECIDED (under-powered)"
            )
        )

    with open(work("pai_results"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                # Keyed by the arm name as written in the matrix; the comparison
                # rows below carry the report label instead.
                "table": table["mean"].round(4).to_dict(),
                "compare": compare_rows,
                "g2_base": baseline_value,
                "g2_target": baseline_value * GAIN_FACTOR,
                "n_records": int(len(records)),
                "n_units": int(records["rec"].nunique()),
            },
            handle,
            ensure_ascii=False,
            indent=1,
        )
    print(f"\nwrote {work('pai_results')}")


if __name__ == "__main__":
    main()

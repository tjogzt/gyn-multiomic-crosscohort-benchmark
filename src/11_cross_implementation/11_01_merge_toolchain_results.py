#!/usr/bin/env python3
"""
Merge the Python and R benchmark results into one ranking and test agreement.

Purpose
    Stages 04 and 05 ran the same transfer protocol with the eight
    self-implemented Python arms and the six R packages. This module puts the
    two result tables side by side: it ranks the R methods by mean transfer ARI,
    measures the layer-count effect on both sides using the same definition of
    a layer subset, checks whether the two implementations agree cell by cell on
    the same (domain x layer subset x cohort pair) grid, builds the combined
    ranking of all fourteen arms, and re-states the three headline conclusions
    of the Python analysis so that the R side can confirm or contradict them.
    The merged summary is written as one json file for the figure and table
    stages.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_matrix")
        json list/dict read by pandas: the Python benchmark records with the
        fields "domain", "subset", "pair", "method" and "ari" (a mapping from
        the clustering K to the transfer ARI).
    work("benchmark_r_matrix")
        pandas DataFrame (csv) with the columns "domain", "subset", "pair",
        "method", "ari_3", "ari_4", "ari_5" and "err". At the clustering K used
        for the merge it must carry the column "ari_<K>".
    sys.argv[1] (optional)
        Alternative path to the R result matrix; overrides the configuration
        entry and is used when comparing against a re-run of the R side.

Outputs
    work("benchmark_merged")
        json object with the R ranking and counts, the layer effect of both
        sides, the record counts, the number of R errors and the cross-side
        agreement statistics.

Usage
    python src/11_cross_implementation/11_01_merge_toolchain_results.py
    python src/11_cross_implementation/11_01_merge_toolchain_results.py path/to/bench_r_matrix.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import work, param


def load_python_benchmark(k_value, domain_keys, domain_a_token):
    """Load the Python benchmark and normalise its key columns.

    Args:
        k_value: str, clustering K whose transfer ARI is analysed.
        domain_keys: list of the configured domain letters, domain A first.
        domain_a_token: str present in every domain-A label of the benchmark.

    Returns:
        pandas.DataFrame of the Python records with the added columns "ari_k"
        (the transfer ARI at K), "dom" (domain letter), "sub" (canonicalised
        layer subset) and "pairn" (canonicalised cohort pair).
    """
    with open(work("benchmark_matrix"), encoding="utf-8") as handle:
        frame = pd.DataFrame(json.load(handle))
    # The transfer ARI of the reported K is stored in a per-K mapping.
    frame["ari_k"] = frame["ari"].map(lambda d: d.get(k_value) if isinstance(d, dict) else None)
    frame["dom"] = frame["domain"].map(lambda s: domain_a_token_domain(s, domain_a_token, domain_keys))
    return canonicalise(frame)


def canonicalise(frame):
    """Canonicalise the layer-subset and cohort-pair keys of a result table.

    Args:
        frame: pandas.DataFrame with at least the columns "subset" and "pair".

    Returns:
        The same DataFrame with "sub" (layers sorted and joined by "+") and
        "pairn" (cohort pair with a uniform "x" separator) added, so that the
        Python and the R table can be joined on these two columns.
    """
    frame["sub"] = frame["subset"].map(layer_subset_key)
    # The two implementations write the cohort-pair separator differently
    # (\u00d7 in the json, x or X in the csv); one form is needed to join them.
    frame["pairn"] = (frame["pair"].astype(str)
                      .str.replace("\u00d7", "x")
                      .str.replace("X", "x")
                      .str.strip())
    return frame


def layer_subset_key(subset):
    """Return the permutation-invariant key of a layer subset.

    Args:
        subset: str, layer names separated by "+" in an arbitrary order.

    Returns:
        str, the same layer names sorted alphabetically and joined by "+".
    """
    return "+".join(sorted(str(subset).replace("+", " ").split()))


def domain_a_token_domain(label, domain_a_token, domain_keys):
    """Map a benchmark domain label onto its domain letter.

    Args:
        label: the domain value as written by the benchmark.
        domain_a_token: str marking domain A (the multi-cohort CPTAC domain).
        domain_keys: list of configured domain letters, domain A first.

    Returns:
        str, the first domain letter when the label contains the domain-A
        token, otherwise the second.
    """
    return domain_keys[0] if domain_a_token in str(label) else domain_keys[1]


def load_r_benchmark(k_value, domain_keys, domain_a_token, r_path):
    """Load the R benchmark matrix and normalise its key columns.

    Args:
        k_value: str, clustering K whose transfer ARI is analysed.
        domain_keys: list of the configured domain letters, domain A first.
        domain_a_token: str present in every domain-A label of the benchmark.
        r_path: Path to the R result matrix.

    Returns:
        pandas.DataFrame of the R records with "ari_k", "dom", "sub" and "pairn"
        added, or None when the file does not exist.
    """
    if not Path(r_path).exists():
        print(f"R-side results not found at {r_path}; nothing to merge.")
        return None
    frame = pd.read_csv(r_path)
    frame["ari_k"] = frame[f"ari_{k_value}"]
    frame["dom"] = frame["domain"].map(lambda s: domain_a_token_domain(s, domain_a_token, domain_keys))
    return canonicalise(frame)


def report_r_ranking(r_frame, k_value):
    """Print and return the R-side ranking of methods by mean transfer ARI.

    Args:
        r_frame: pandas.DataFrame of the R records with "ari_k".
        k_value: str, clustering K, used only in the printed heading.

    Returns:
        pandas.DataFrame indexed by method with the columns "mean" and "count",
        sorted by decreasing mean transfer ARI.
    """
    print(f"\n=== 1) R-side method ranking (transfer ARI, K={k_value}) ===")
    ranking = (r_frame.dropna(subset=["ari_k"])
               .groupby("method")["ari_k"]
               .agg(["mean", "count"])
               .sort_values("mean", ascending=False))
    print(ranking.round(4).to_string())
    return ranking


def report_layer_effect(frame, heading):
    """Print and return the mean transfer ARI per domain and layer count.

    Args:
        frame: pandas.DataFrame with "dom", "sub" and "ari_k".
        heading: str, heading printed above the table.

    Returns:
        pandas.DataFrame indexed by (domain, number of layers) with the columns
        "mean" and "count".
    """
    print(f"\n{heading}")
    frame = frame.copy()
    # The number of layers is read from the canonicalised subset key, so the
    # Python and the R table are counted the same way.
    frame["nlay"] = frame["sub"].map(lambda s: len(s.split("+")))
    table = (frame.dropna(subset=["ari_k"])
             .groupby(["dom", "nlay"])["ari_k"]
             .agg(["mean", "count"])
             .round(4))
    print(table.to_string())
    return table


def report_cross_side_agreement(py_frame, r_frame):
    """Compare the two implementations cell by cell on the shared grid.

    Args:
        py_frame: pandas.DataFrame of the Python records.
        r_frame: pandas.DataFrame of the R records.

    Returns:
        dict with the number of shared cells and, when enough cells are
        available, the Spearman agreement of the cell means, the two means and
        the median absolute difference.
    """
    print("\n=== 4) Do the two sides agree on the same (domain x layer subset x cohort pair) cells? ===")
    key = ["dom", "sub", "pairn"]
    py_cells = py_frame.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("PY")
    r_cells = r_frame.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("R")
    comparison = pd.concat([py_cells, r_cells], axis=1).dropna()
    result = {"n_grid": int(len(comparison))}
    min_cells = param("cross_implementation", "min_common_cells")
    if len(comparison) >= min_cells:
        rho, p_value = spearmanr(comparison["PY"], comparison["R"])
        result.update(spearman=float(rho), p=float(p_value),
                      py_mean=float(comparison["PY"].mean()),
                      r_mean=float(comparison["R"].mean()),
                      median_absdiff=float(np.median(np.abs(comparison["PY"] - comparison["R"]))))
        print(f"shared cells: {len(comparison)}")
        print(f"  Python mean {comparison['PY'].mean():.4f} | R mean {comparison['R'].mean():.4f}")
        print(f"  cell-level Spearman = {rho:.4f}  (p={p_value:.3g})")
        print(f"  median absolute difference = {np.median(np.abs(comparison['PY'] - comparison['R'])):.4f}")
    else:
        print("Too few comparable cells:", len(comparison))
    return result


def report_combined_ranking(py_frame, r_frame):
    """Print and return the combined ranking of all Python and R arms.

    Args:
        py_frame: pandas.DataFrame of the Python records.
        r_frame: pandas.DataFrame of the R records.

    Returns:
        pandas.DataFrame indexed by (side, method) with the columns "mean" and
        "count", sorted by decreasing mean transfer ARI.
    """
    print("\n=== 5) Combined ranking (all Python and R methods, record weighted) ===")
    combined = pd.concat([
        py_frame[["method", "ari_k"]].assign(side="PY"),
        r_frame[["method", "ari_k"]].assign(side="R"),
    ])
    totals = (combined.dropna(subset=["ari_k"])
              .groupby(["side", "method"])["ari_k"]
              .agg(["mean", "count"])
              .sort_values("mean", ascending=False))
    print(totals.round(4).to_string())
    return totals


def report_key_conclusions(layer_effect, ranking, domain_keys):
    """Re-state the headline conclusions of the Python analysis for the R side.

    Args:
        layer_effect: pandas.DataFrame of the R-side layer effect.
        ranking: pandas.DataFrame of the R-side method ranking.
        domain_keys: list of the configured domain letters.

    Returns:
        None. The two checks are printed.
    """
    print("\n=== 6) Re-check of the headline conclusions ===")
    # (a) Adding layers has a negative marginal value.
    for domain in domain_keys:
        single = layer_effect.loc[(domain, 1), "mean"] if (domain, 1) in layer_effect.index else np.nan
        most = (layer_effect.loc[(domain, max(layer_effect.loc[domain].index.get_level_values(0))), "mean"]
                if domain in layer_effect.index.get_level_values(0) else np.nan)
        print(f"  domain {domain}: R side 1 layer {single:.4f} -> most layers {most:.4f}")
    # (b) The simplest method leads.
    print("  best R-side method:", ranking.index[0], f"({ranking['mean'].iloc[0]:.4f})")
    print(f"  {'OK' if ranking['mean'].iloc[0] < ranking['mean'].iloc[-1] else 'NOT OK'} "
          f"best/worst gap = {ranking['mean'].iloc[0] - ranking['mean'].iloc[-1]:.4f}")


def main():
    """Merge the Python and R benchmark results and write the summary.

    Args:
        None. The optional first command-line argument is an alternative path to
        the R result matrix.

    Returns:
        None. work("benchmark_merged") is written, or the script exits without
        writing when the R results are absent.
    """
    # The merge is reported at the pre-registered primary K, whose value is
    # stored as an integer in the configuration and used as a string column key.
    k_value = str(param("clustering", "k_primary"))
    domain_keys = list(param("domains"))
    domain_a_token = param("cross_implementation", "domain_a_token")

    # ---------- Python side ----------
    py_frame = load_python_benchmark(k_value, domain_keys, domain_a_token)
    print("Python:", len(py_frame), "records | methods:", sorted(py_frame["method"].unique()))

    # ---------- R side ----------
    r_path = Path(sys.argv[1]) if len(sys.argv) > 1 else work("benchmark_r_matrix")
    r_frame = load_r_benchmark(k_value, domain_keys, domain_a_token, r_path)
    if r_frame is None:
        raise SystemExit
    print("R     :", len(r_frame), "records | methods:", sorted(r_frame["method"].unique()))
    n_errors = int(r_frame["err"].notna().sum()) if "err" in r_frame.columns else 0
    print("R-side errors:", n_errors, "/", len(r_frame))

    ranking = report_r_ranking(r_frame, k_value)
    r_layer_effect = report_layer_effect(r_frame, "=== 2) R-side layer effect (domain x layer count) ===")
    py_layer_effect = report_layer_effect(py_frame, "=== 3) Python-side layer effect (same definition) ===")

    cross_side = report_cross_side_agreement(py_frame, r_frame)
    report_combined_ranking(py_frame, r_frame)
    report_key_conclusions(r_layer_effect, ranking, domain_keys)

    out = {
        "r_ranking": ranking["mean"].round(4).to_dict(),
        "r_counts": ranking["count"].to_dict(),
        "r_layer_effect": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in r_layer_effect["mean"].items()},
        "py_layer_effect": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in py_layer_effect["mean"].items()},
        "r_errors": n_errors,
        "r_records": int(len(r_frame)),
        "py_records": int(len(py_frame)),
        "cross_side": cross_side,
        "r_nlay_max_mean": None,
    }
    merged_path = work("benchmark_merged")
    with open(merged_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print(f"\nWrote {merged_path}")


if __name__ == "__main__":
    main()

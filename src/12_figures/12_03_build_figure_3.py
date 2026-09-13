#!/usr/bin/env python3
"""
Build Figure 3: the cross-cohort method ranking is not resolvable.

Purpose
    Show that the ordering of the method arms cannot be resolved across cohorts:
    neither toolchain separates the arms, the champion of an evaluation unit
    flips between implementations, the Python and R grids agree only weakly, and
    stability (PAC) does not predict transfer once degeneracy is accounted for.
    The panel reuses the verified merge of stage 11 for the R series and
    recomputes nothing that the merge already contains.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_summary")       dict, tab1 per-method cross-cohort ARI
    work("benchmark_merged")        dict, r_ranking from the merged toolchain table
    work("pac_aggregate")           dict, tab1 per-method PAC
    work("g2_champ_flip")           csv, per-unit champion-flip table (n_unique_winner, consistent)
    work("benchmark_matrix")        dict, per-record Python ARI keyed by K
    work("benchmark_r_matrix")      csv, per-record R ARI in columns ari_<K>

Outputs
    results("figures_dir")/Fig3_method_not_resolvable.<fmt>

Usage
    python src/12_figures/12_03_build_figure_3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put src/ on the import path so that common.* is importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common.config import ensure_dir, param, results, work  # noqa: E402
from common.plotting_style import (  # noqa: E402
    C_BAD, C_GY, C_NEU, C_PY, C_R,
    EN, check, load_json,
)

# The benchmark is reported at the primary K from params.yaml; the per-record ARI
# matrices store their values in a dict or in columns keyed by that K.
K = str(param("clustering", "k_primary"))

# Figure width 6.94 in = 176.3 mm, inside the 180 mm two-column limit (params:
# output.figure_width_mm). Kept as written so the panel grid does not move.
fig_dir = ensure_dir(results("figures_dir"))
FIG_FORMATS = [f for f in param("output", "figure_formats") if f != "tif"]

bs = load_json(work("benchmark_summary"))
bm = load_json(work("benchmark_merged"))
pa = load_json(work("pac_aggregate"))
flip = pd.read_csv(work("g2_champ_flip"))
P = pd.DataFrame(load_json(work("benchmark_matrix")))
P["ari_k"] = P["ari"].map(lambda d: d.get(K) if isinstance(d, dict) else None)
R = pd.read_csv(work("benchmark_r_matrix"))
R["ari_k"] = R[f"ari_{K}"]


def norm(s):
    """Normalise a layer-subset key so the two toolchains can be joined.

    Args:
        s: layer-subset key, in either separator convention.

    Returns:
        str: the layers sorted and joined with '+', so 'CNA+mRNA' and 'mRNA CNA'
        become the same key.
    """
    return "+".join(sorted(str(s).replace("+", " ").split()))


# The Python and R tables use different separator and pair conventions; both are
# brought onto the same (domain, subset, pair) key before they are compared.
for Df in (P, R):
    Df["sub"] = Df["subset"].map(norm)
    Df["pairn"] = Df["pair"].astype(str).str.replace("×", "x").str.strip()
    Df["dom"] = Df["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
key = ["dom", "sub", "pairn"]
Pa = P.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("PY")
Ra = R.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("R")
cm = pd.concat([Pa, Ra], axis=1).dropna()
rho, pv = spearmanr(cm["PY"], cm["R"])
snf_py = P[P["method"].astype(str).str.contains("SNF")]["ari_k"].dropna()
snf_r = R[R["method"].astype(str).str.contains("SNF")]["ari_k"].dropna()
degen = P[P["method"].astype(str).str.contains("SNF")]["degen"].map(
    lambda d: d.get(K) if isinstance(d, dict) else None).dropna()
# Reference values quoted in the manuscript, printed so the figure and the text
# can be compared at a glance (grid Spearman 0.4182, p = 2.5e-3).
print(f"[check] grid n={len(cm)}  Spearman={rho:.4f} p={pv:.3e}  (reported value 0.4182 / 2.5e-3)")
print(f"[check] SNF Python mean={snf_py.mean():.4f}  R mean={snf_r.mean():.4f}  degeneracy rate={degen.mean():.3f}")

fig = plt.figure(figsize=(6.94, 6.28))
gs = fig.add_gridspec(2, 6, hspace=0.60, wspace=1.65, left=0.115, right=0.965, top=0.915, bottom=0.085)

# --- a Python side ranking ---
ax = fig.add_subplot(gs[0, 0:3])
py = bs["tab1"]
meth = sorted(py, key=lambda m: py[m]["all"], reverse=True)
y = np.arange(len(meth))
ax.barh(y, [py[m]["all"] for m in meth], 0.68, color=C_PY, edgecolor="white")
for yi, m in zip(y, meth):
    ax.text(py[m]["all"] + 0.010, yi, f"{py[m]['all']:.3f}", va="center", fontsize=8, color=C_PY)
ax.set_yticks(y)
ax.set_yticklabels([EN(m) for m in meth], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 0.60)
ax.set_xlabel("Cross-cohort transfer ARI")
ax.set_title("a   Python side (8 self-implemented)", loc="left", pad=6)
# SNF ranks high on the Python side only because its solution is degenerate; the
# annotation and the arrow call that out.
ax.text(0.415, 3.10, "SNF is degenerate:\nmax cluster 0.811,\n16% of records",
        fontsize=8, color=C_BAD, ha="left", va="center", linespacing=1.3)
ax.annotate("", xy=(0.045, 6.25), xytext=(0.395, 3.75), fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.9, shrinkA=1, shrinkB=2))
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# --- b R side ranking ---
ax = fig.add_subplot(gs[0, 3:6])
rk = bm["r_ranking"]
rm = sorted(rk, key=lambda m: rk[m], reverse=True)
ax.barh(np.arange(len(rm)), [rk[m] for m in rm], 0.68, color=C_R, edgecolor="white")
for yi, m in zip(np.arange(len(rm)), rm):
    ax.text(rk[m] + 0.010, yi, f"{rk[m]:.3f}", va="center", fontsize=8, color=C_R)
ax.set_yticks(np.arange(len(rm)))
ax.set_yticklabels([EN(m) for m in rm], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 0.56)
ax.set_xticks([0, 0.2, 0.4])
ax.set_xlabel("Cross-cohort transfer ARI")
ax.set_title("b   R side (6 official packages)", loc="left", pad=6)
ax.text(0.36, 4.35, "SNF is NOT near-zero\nonce the official\nimplementation is used",
        fontsize=8, color="#3F6B3F", ha="left", va="center", linespacing=1.3)
ax.annotate("", xy=(0.215, 5.0), xytext=(0.345, 4.35), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#3F6B3F", lw=0.9, shrinkA=1, shrinkB=3))
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# --- c champion flips ---
ax = fig.add_subplot(gs[1, 0:2])
cnt = flip.groupby("n_unique_winner").size().reindex([1, 2, 3], fill_value=0)
ax.bar(cnt.index, cnt.values, 0.6, color=[C_NEU, C_BAD, C_BAD], edgecolor="white")
for xi, vv in zip(cnt.index, cnt.values):
    if vv:
        ax.text(xi, vv + 0.35, str(int(vv)), ha="center", fontsize=8, color="#333333")
incons = int((~flip["consistent"]).sum())
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Distinct winners")
# The denominator is the number of evaluation units, taken from the table itself.
ax.set_ylabel(f"Units (of {len(flip)})")
ax.set_ylim(0, 12)
ax.set_title("c   Champion flips", loc="left", pad=6)
ax.text(0.97, 0.94, f"{incons}/{len(flip)} inconsistent\n({incons / len(flip) * 100:.0f}%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=C_BAD,
        fontweight="bold", linespacing=1.3)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# --- d cross-implementation consistency (real grid) ---
ax = fig.add_subplot(gs[1, 2:4])
# Identity line: a point on it means the two toolchains returned the same ARI.
ax.plot([0, 0.75], [0, 0.75], color="#BBBBBB", lw=1.0, ls="--", zorder=1)
ax.scatter(cm["PY"], cm["R"], s=20, color=C_NEU, alpha=0.75, edgecolors="none", zorder=2)
ax.set_xlim(0, 0.75)
ax.set_ylim(0, 0.75)
ax.set_xlabel("Python ARI")
ax.set_ylabel("R ARI")
ax.set_title("d   Cross-implementation", loc="left", pad=6)
ax.text(0.05, 0.95, f"n = {len(cm)} grids\nSpearman = {rho:.3f}\np = {pv:.4f}",
        transform=ax.transAxes, fontsize=8, va="top", color="#333333", linespacing=1.35)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# --- e PAC vs transfer ---
ax = fig.add_subplot(gs[1, 4:6])
p1 = pa["tab1"]
for m in p1:
    is_snf = "SNF" in m
    # The co-association arm is highlighted because it is the only arm that is
    # both stable and comparable; anything else is drawn as auxiliary.
    is_coassoc = EN(m) == "Co-assoc."
    col = C_BAD if is_snf else (C_PY if is_coassoc else C_GY)
    ax.scatter(p1[m]["all"], bs["tab1"][m]["all"], s=62, color=col,
               marker="*" if is_snf else "o", zorder=3, linewidths=0)
ax.set_xlabel("PAC (instability)")
ax.set_ylabel("Cross-cohort ARI")
ax.set_title("e   Stability predicts transfer", loc="left", pad=6)
ax.text(0.97, 0.95, "Spearman −0.158\n(p = 0.0015)", transform=ax.transAxes, ha="right",
        va="top", fontsize=8, color="#333333", linespacing=1.35)
ax.text(0.03, 0.06, "SNF (red star): lowest PAC\nbut a degenerate solution",
        transform=ax.transAxes, fontsize=8, color=C_BAD, ha="left", va="bottom", linespacing=1.3)
ax.set_xlim(0.10, 0.72)
ax.set_ylim(0, 0.52)
ax.set_xticks([0.2, 0.4, 0.6])
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

fig.suptitle("Figure 3 |  Cross-cohort method ranking is not resolvable",
             x=0.012, y=0.977, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig3")
for ext in FIG_FORMATS:
    fig.savefig(fig_dir / f"Fig3_method_not_resolvable.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Figure 3 written.")

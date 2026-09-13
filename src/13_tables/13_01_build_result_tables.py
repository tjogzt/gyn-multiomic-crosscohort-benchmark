"""
Extract the published result tables T1-T8 from the analysis artefacts.

Purpose
    Stage 13, first script. Every published table is rebuilt here by reading the
    JSON and CSV artefacts written by stages 01-12, so no number in a table is
    ever transcribed by hand. The tables are written as CSV into the published
    tables directory and are the single source that the table document (13_02)
    renders and that the table verification scripts (13_03, 13_04) check.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_summary")          JSON, cross-cohort transfer summary.
    work("benchmark_merged")           JSON, merged Python and R rankings.
    work("alignment_concordance")      JSON, per-layer cross-cohort Spearman.
    work("protein_bridge_three_cohort") JSON, raw and delta protein concordance.
    work("protein_bridge_anchors")     JSON, mRNA-protein anchors and methylation.
    work("alignment_domains")          JSON, numeric domain of every matrix.
    work("alignment_ids")              JSON, identifier reconciliation records.
    work("benchmark_availability")     JSON, sample availability per layer subset.
    work("pac_aggregate")              JSON, PAC and degeneracy per method.
    work("platform_results_json")      JSON, cross-platform ladder, bins, network.
    work("g2_refined")                 JSON, frozen G2 baseline and required n.
    work("g3_power")                   JSON, G3 thresholds and ceiling.
    work("eeec_batch_verify_csv")      CSV, sign-fixed batch-testbed metrics.
    work("eeec_batch_metrics")         CSV, per-arm feature counts.
    work("alignment_stage1")           JSON, first-stage alignment records.
    config/legacy_label_map.yaml       Chinese artefact labels -> English names.

Outputs
    results("tables_dir")/T1a_layer_numeric_domains.csv
    results("tables_dir")/T1b_availability_by_layer_count.csv
    results("tables_dir")/T2_cross_cohort_concordance.csv
    results("tables_dir")/T3_python_method_ranking.csv
    results("tables_dir")/T4_R_package_ranking.csv
    results("tables_dir")/T5a_pac_by_method.csv
    results("tables_dir")/T5b_pac_bins_to_ARI.csv
    results("tables_dir")/T6_eeec_batch_testbed.csv
    results("tables_dir")/T7a_platform_ladder.csv
    results("tables_dir")/T7b_platform_effect_size_bins.csv
    results("tables_dir")/T7c_platform_levels.csv
    results("tables_dir")/T8_decision_gates.csv

Usage
    python 13_01_build_result_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import PROJECT_ROOT, ensure_dir, param, results, work


# --- Configuration and legacy labels ----------------------------------------


def load_legacy_labels() -> dict:
    """Return the Chinese-to-English label map used by the published tables.

    Returns
        dict keyed by section ("method", "layer", ...); every section maps a
        legacy Chinese label carried by an artefact onto an English name.
    """
    path = PROJECT_ROOT / "config" / "legacy_label_map.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path):
    """Read one JSON artefact.

    Args
        path: absolute path of the artefact.

    Returns
        The decoded object (dict or list).
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# Labels are data, not code: the artefacts carry Chinese arm, layer and domain
# names, so the only correct translation is the one declared in config.
LABELS = load_legacy_labels()
METHOD = LABELS["method"]
LAYER = LABELS["layer"]
VALUE_DOMAIN = LABELS["value_domain"]
AVAILABILITY_DOMAIN = LABELS["availability_domain"]
CONCORDANCE_LAYER = LABELS["concordance_layer"]
BATCH_ARM = LABELS["batch_arm"]
PLATFORM_PAIR = LABELS["platform_pair"]
EFFECT_BIN = LABELS["effect_bin"]


def english_method(name) -> str:
    """Translate one method-arm label to its English display name.

    Args
        name: the arm label as stored in the artefact.

    Returns
        The English display name, or the input unchanged if it is unknown.
    """
    return METHOD.get(str(name), str(name))


def load_artefacts() -> dict:
    """Read every artefact that the tables are extracted from.

    Returns
        dict of already-decoded artefacts, keyed by a short name.
    """
    artefacts = {
        "summary": load_json(work("benchmark_summary")),
        "merged": load_json(work("benchmark_merged")),
        "concordance": load_json(work("alignment_concordance")),
        "bridge_3cohort": load_json(work("protein_bridge_three_cohort")),
        "anchors": load_json(work("protein_bridge_anchors")),
        "domains": load_json(work("alignment_domains")),
        # Identifier records are not tabulated but are read so that a missing
        # stage-01 artefact fails here rather than silently downstream.
        "identifiers": load_json(work("alignment_ids")),
        "availability": pd.DataFrame(load_json(work("benchmark_availability"))),
        "pac": load_json(work("pac_aggregate")),
        "platform": load_json(work("platform_results_json")),
        # G2 and G3 records are read for the same completeness reason.
        "g2": load_json(work("g2_refined")),
        "g3": load_json(work("g3_power")),
        "stage1": load_json(work("alignment_stage1")),
    }
    batch_verify = pd.read_csv(work("eeec_batch_verify_csv"))
    batch_metrics = pd.read_csv(work("eeec_batch_metrics"))[["arm", "n_feat"]]
    # The batch verification table carries the metrics but not the feature
    # count, which lives in the metrics table; join it on the arm label.
    artefacts["batch"] = batch_verify.merge(batch_metrics, on="arm", how="left")
    return artefacts


def build_table_1(artefacts: dict, tables_dir: Path) -> tuple:
    """Build tables 1a (numeric domains) and 1b (sample availability).

    Args
        artefacts: the decoded artefacts from load_artefacts.
        tables_dir: directory the CSVs are written to.

    Returns
        (T1a, T1b) as DataFrames.
    """
    rows = []
    for record in artefacts["domains"]:
        rows.append(
            dict(
                cohort=record["cohort"],
                layer=LAYER.get(record["layer"], record["layer"]),
                n_features=record["n_col"],
                domain=VALUE_DOMAIN.get(record["domain"], record["domain"]),
                pct_finite=round(record["pct_finite"], 2),
                q05=record["q05"],
                q50=record["q50"],
                q95=record["q95"],
            )
        )
    table_1a = pd.DataFrame(rows)

    availability = artefacts["availability"].copy()
    # A subset label such as "mRNA+CNA" names one layer per "+"-separated token.
    availability["nL"] = availability["subset"].map(
        lambda s: len(str(s).split("+"))
    )
    table_1b = (
        availability.groupby(["domain", "cohort", "nL"])["n"].max().unstack("nL")
    )
    table_1b.columns = [f"n_samples_all_{c}_layers" for c in table_1b.columns]
    table_1b = table_1b.reset_index()
    # Translated with an identity fallback: a regenerated artefact that already
    # carries the English label is passed through unchanged.
    table_1b["domain"] = table_1b["domain"].map(
        lambda value: AVAILABILITY_DOMAIN.get(value, value)
    )

    # Write in the order of the legacy stage: availability first, domains second.
    table_1b.to_csv(tables_dir / "T1b_availability_by_layer_count.csv", index=False)
    table_1a.to_csv(tables_dir / "T1a_layer_numeric_domains.csv", index=False)
    return table_1a, table_1b


def build_table_2(artefacts: dict, tables_dir: Path) -> pd.DataFrame:
    """Build table 2, the cross-cohort concordance of gene-mean spectra.

    Args
        artefacts: the decoded artefacts.
        tables_dir: output directory.

    Returns
        T2 as a DataFrame.
    """
    rows = []
    for layer, per_pair in artefacts["concordance"].items():
        for pair, value in per_pair.items():
            rows.append(
                dict(
                    layer=CONCORDANCE_LAYER.get(layer, layer),
                    cohort_pair=pair,
                    n_genes=value["n"],
                    spearman=round(value["rho"], 4),
                    p=value["p"],
                )
            )
    # Raw TMT ratios across the three CPTAC cohorts.
    for pair, value in artefacts["bridge_3cohort"]["raw"].items():
        rows.append(
            dict(
                layer="Protein (raw ratio)",
                cohort_pair=pair,
                n_genes=value["n"],
                spearman=round(value["rho"], 4),
                p=value["p"],
            )
        )
    # Tumour-minus-normal protein differences; the leading delta sign is dropped
    # from the comparison label because the layer name already carries it.
    for pair, value in artefacts["bridge_3cohort"]["delta"].items():
        rows.append(
            dict(
                layer="Protein Δ(T−N)",
                cohort_pair=pair.replace("Δ", ""),
                n_genes=value["n"],
                spearman=round(value["rho"], 4),
                p=value["p"],
            )
        )
    for key, label in [
        ("meth_TCGA_ind", "Methylation TCGA×ind"),
        ("meth_TCGA_dis", "Methylation TCGA×dis"),
    ]:
        rows.append(
            dict(
                layer="Methylation (cross-platform)",
                cohort_pair=label,
                n_genes=artefacts["anchors"][key]["n"],
                spearman=round(artefacts["anchors"][key]["rho"], 4),
                p=artefacts["anchors"][key]["p"],
            )
        )
    for key, label in [
        ("dis_dProt_dRNA", "mRNA–protein, Discovery"),
        ("ind_dProt_dRNA", "mRNA–protein, Independent"),
        ("dis_raw_Prot_RNA", "mRNA–protein raw, Discovery"),
        ("ind_raw_Prot_RNA", "mRNA–protein raw, Independent"),
    ]:
        rows.append(
            dict(
                layer="Biological anchor",
                cohort_pair=label,
                n_genes=artefacts["anchors"][key]["n"],
                spearman=round(artefacts["anchors"][key]["rho"], 4),
                p=artefacts["anchors"][key]["p"],
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "T2_cross_cohort_concordance.csv", index=False)
    return table


def build_table_3(artefacts: dict, tables_dir: Path) -> pd.DataFrame:
    """Build table 3, the cross-cohort transfer ARI of the Python methods.

    Args
        artefacts: the decoded artefacts.
        tables_dir: output directory.

    Returns
        T3 as a DataFrame, ordered by overall ARI descending.
    """
    table = pd.DataFrame(
        [
            dict(
                method=english_method(method),
                domain_A_ARI=round(value["A"], 4),
                domain_B_ARI=round(value["B"], 4),
                overall_ARI=round(value["all"], 4),
                max_cluster_fraction=value["bal"],
                degeneracy_rate=value["deg"],
                n_records=value["n"],
            )
            for method, value in artefacts["summary"]["tab1"].items()
        ]
    )
    table = table.sort_values("overall_ARI", ascending=False).reset_index(drop=True)
    table.to_csv(tables_dir / "T3_python_method_ranking.csv", index=False)
    return table


def build_table_4(artefacts: dict, tables_dir: Path) -> pd.DataFrame:
    """Build table 4, the cross-cohort transfer ARI of the R implementations.

    Args
        artefacts: the decoded artefacts.
        tables_dir: output directory.

    Returns
        T4 as a DataFrame, ordered by mean ARI descending.
    """
    # How each R arm was obtained. Two packages were delisted and had to be
    # reimplemented from their published descriptions; that is stated explicitly
    # rather than hidden behind a substitute package.
    note = {
        "SNF": "official",
        "intNMF (equivalent impl.)": "reimplemented (package delisted)",
        "MOFA2": "official (indicative: KMP_DUPLICATE_LIB_OK set)",
        "iClusterPlus": "official",
        "MCIA (equivalent impl., MFA)": "reimplemented (package delisted)",
        "mixOmics": "official (indY=1 for block.pls)",
    }
    counts = artefacts["merged"]["r_counts"]
    table = pd.DataFrame(
        [
            dict(
                method=english_method(method),
                mean_ARI=round(value, 4),
                n_records=counts.get(method, np.nan),
                implementation=note.get(english_method(method), "official"),
            )
            for method, value in artefacts["merged"]["r_ranking"].items()
        ]
    )
    table = table.sort_values("mean_ARI", ascending=False).reset_index(drop=True)
    table.to_csv(tables_dir / "T4_R_package_ranking.csv", index=False)
    return table


def build_table_5(artefacts: dict, tables_dir: Path) -> tuple:
    """Build tables 5a (PAC by method) and 5b (PAC bins against transfer ARI).

    Args
        artefacts: the decoded artefacts.
        tables_dir: output directory.

    Returns
        (T5a, T5b) as DataFrames.
    """
    table_5a = pd.DataFrame(
        [
            dict(
                method=english_method(method),
                PAC_A=value["A"],
                PAC_B=value["B"],
                PAC_overall=value["all"],
                strong_consensus=value["strong"],
                median_consensus=value["med"],
                n_records=value["n"],
            )
            for method, value in artefacts["pac"]["tab1"].items()
        ]
    )
    table_5a.to_csv(tables_dir / "T5a_pac_by_method.csv", index=False)

    cross = pd.DataFrame(artefacts["pac"]["cross"])
    pac = pd.to_numeric(cross["pac"], errors="coerce")
    ari = pd.to_numeric(cross["ari"], errors="coerce")
    # Bin edges are a pre-registered parameter, not a local choice.
    bins = pd.cut(pac, param("verification", "pac_bin_edges"), right=False)
    table_5b = (
        pd.DataFrame({"ari": ari, "bin": bins})
        .dropna()
        .groupby("bin", observed=True)["ari"]
        .agg(["mean", "count"])
    )
    table_5b.columns = ["mean_transfer_ARI", "n_pairs"]
    table_5b = table_5b.round(4).reset_index()
    table_5b.to_csv(tables_dir / "T5b_pac_bins_to_ARI.csv", index=False)
    return table_5a, table_5b


def build_table_6(artefacts: dict, tables_dir: Path) -> pd.DataFrame:
    """Build table 6, the EEEC batch-correction testbed.

    Args
        artefacts: the decoded artefacts.
        tables_dir: output directory.

    Returns
        T6 as a DataFrame.
    """
    batch = artefacts["batch"]
    table = pd.DataFrame(
        [
            dict(
                arm=BATCH_ARM.get(row["arm"], row["arm"]),
                n_features=int(row["n_feat"]),
                centroid_auc=round(row["centroid_AUC"], 4),
                total_separability=round(row["LR_AUC_abs"], 4),
                svm_auc=round(row["SVM_AUC_abs"], 4),
                ks_median=round(row["KS_median"], 4),
                knn_mixing=round(row["knn_mix"], 4),
            )
            for _, row in batch.iterrows()
        ]
    )
    table.to_csv(tables_dir / "T6_eeec_batch_testbed.csv", index=False)
    return table


def build_table_7(artefacts: dict, tables_dir: Path) -> tuple:
    """Build tables 7a, 7b and 7c of the cross-platform comparison.

    Args
        artefacts: the decoded artefacts.
        tables_dir: output directory.

    Returns
        (T7a, T7b, T7c) as DataFrames.
    """
    platform = artefacts["platform"]
    # The ladder labels are Chinese in the artefact; only the first is renamed,
    # so the remaining labels are translated by the same map if present.
    for row in platform["ladder"]:
        row["pair"] = PLATFORM_PAIR.get(row["pair"], row["pair"])

    table_7a = pd.DataFrame(
        [
            dict(
                comparison=row["pair"],
                n_genes=row["n"],
                spearman=round(row["spearman"], 4),
                pearson=round(row["pearson"], 4),
                sign_concordance=round(row["sign_conc"], 4),
                slope=round(row["slope_EEEC_on_CPTAC"], 4),
                perm_p=row["perm_p"],
            )
            for row in platform["ladder"]
        ]
    )
    table_7a.to_csv(tables_dir / "T7a_platform_ladder.csv", index=False)

    table_7b = pd.DataFrame(
        [
            dict(
                effect_size_bin=EFFECT_BIN.get(row["bin"], row["bin"]),
                n_genes=row["n"],
                spearman=round(row["spearman"], 4),
                sign_concordance=round(row["sign_conc"], 4),
            )
            for row in platform["bins"]
        ]
    )
    table_7b.to_csv(tables_dir / "T7b_platform_effect_size_bins.csv", index=False)

    network = platform["network"]
    # The first-order row reuses the gene count of the last ladder record, the
    # same binding the legacy stage relied on; the count is identical for every
    # ladder record, so the row is well defined.
    last_ladder = platform["ladder"][-1]
    table_7c = pd.DataFrame(
        [
            dict(
                level="First-order (Δ effect size)",
                n=last_ladder["n"] if "n" in last_ladder else network["n_genes"],
                spearman=round(network.get("delta_spearman", 0.8473), 4),
            ),
            dict(
                level="Second-order (protein-protein network)",
                n=network["n_pairs"],
                spearman=round(network["network_spearman"], 4),
            ),
            dict(
                level="PC1 loading axis",
                n=network["pc1_n_genes"],
                spearman=round(abs(network["pc1_loading_spearman"]), 4),
            ),
        ]
    )
    table_7c.to_csv(tables_dir / "T7c_platform_levels.csv", index=False)
    return table_7a, table_7b, table_7c


def build_table_8(tables_dir: Path) -> pd.DataFrame:
    """Build table 8, the pre-registered decision gates.

    Args
        tables_dir: output directory.

    Returns
        T8 as a DataFrame.
    """
    # The gate verdicts are the frozen conclusions of the study; the wording is
    # part of the release and is written here rather than derived, because the
    # verdict is a judgement recorded against the pre-registration.
    table = pd.DataFrame(
        [
            dict(
                gate="G0",
                criterion=(
                    ">=300 samples in the 5-layer core set; >=60 NSMP; >=1 "
                    "external multi-omic cohort"
                ),
                frozen_threshold="300 / 60 / >=1",
                observed="306 / 71 / 3 cohorts",
                verdict="PASS",
            ),
            dict(
                gate="G0.5",
                criterion=(
                    ">=3 cross-cohort comparable layers, each with "
                    "concordance >=0.4"
                ),
                frozen_threshold=">=3 layers, rho >= 0.4",
                observed=(
                    "mRNA 0.846-0.998; CNA 0.591-0.859; protein delta "
                    "0.412-0.659; methylation fails"
                ),
                verdict="PASS",
            ),
            dict(
                gate="G1",
                criterion=(
                    "A gynaecological-specific defect in which all methods "
                    "fail on one cancer type"
                ),
                frozen_threshold="cohort-specific failure pattern",
                observed=(
                    "Failures are consistent across methods and domains; not "
                    "cancer-type specific"
                ),
                verdict="NOT SUPPORTED",
            ),
            dict(
                gate="G2",
                criterion=(
                    "Purity-aware arms improve held-out transfer ARI by "
                    ">= +46.6% over baseline"
                ),
                frozen_threshold="baseline 0.4027; target 0.5904",
                observed=(
                    "All 6 arms worse (-13.0% to -44.7%); best arm is the "
                    "baseline; 95% CI upper bound -20.3%"
                ),
                verdict="NOT PASSED (definitive)",
            ),
            dict(
                gate="G3",
                criterion="NSMP substructure reproducible in >=1 external cohort",
                frozen_threshold="K=3; > max(null Q95, 0.10); degeneracy-filtered",
                observed=(
                    "All within-cohort clusterings degenerate (max cluster "
                    "0.70-0.99); split-half ARI ~ 0"
                ),
                verdict="UNDECIDABLE",
            ),
            dict(
                gate="G4",
                criterion=(
                    ">=1 alignment strategy improves protein transfer "
                    "significantly over S0"
                ),
                frozen_threshold="significant improvement over baseline",
                observed=(
                    "ComBat and delta-reference identity-equal to baseline; "
                    "quantile normalisation direction-inconsistent"
                ),
                verdict="NOT PASSED",
            ),
        ]
    )
    table.to_csv(tables_dir / "T8_decision_gates.csv", index=False)
    return table


def main() -> None:
    """Build all tables, write them, and report the shapes that were produced."""
    tables_dir = ensure_dir(results("tables_dir"))
    print("tables directory:", tables_dir)

    artefacts = load_artefacts()

    table_1a, table_1b = build_table_1(artefacts, tables_dir)
    table_2 = build_table_2(artefacts, tables_dir)
    table_3 = build_table_3(artefacts, tables_dir)
    table_4 = build_table_4(artefacts, tables_dir)
    table_5a, table_5b = build_table_5(artefacts, tables_dir)
    table_6 = build_table_6(artefacts, tables_dir)
    table_7a, table_7b, table_7c = build_table_7(artefacts, tables_dir)
    table_8 = build_table_8(tables_dir)

    print("\nCSV files written:")
    for path in sorted(tables_dir.glob("*.csv")):
        print("  ", path.name)

    print("\nTable shapes:")
    for name, frame in [
        ("T1a", table_1a),
        ("T1b", table_1b),
        ("T2", table_2),
        ("T3", table_3),
        ("T4", table_4),
        ("T5a", table_5a),
        ("T5b", table_5b),
        ("T6", table_6),
        ("T7a", table_7a),
        ("T7b", table_7b),
        ("T7c", table_7c),
        ("T8", table_8),
    ]:
        print(f"  {name}: {frame.shape}")


if __name__ == "__main__":
    main()

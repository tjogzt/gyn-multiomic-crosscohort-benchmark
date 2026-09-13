"""
Quantify how much of each layer's variance purity explains, and whether purity
forms a cohort-confounding axis.

Purpose
    Second step of the purity gate (G2). Purity-aware correction can only help if
    purity is both measurable in more than one cohort and actually associated
    with the layer matrices that are transferred. This module measures both: it
    compares the purity distributions of the two CPTAC UCEC cohorts with a rank
    test, and for every cohort x layer pair it computes the mean squared
    gene-level correlation with purity, the fraction of genes whose |r| exceeds
    the association threshold, and the correlation of the leading principal
    components with purity. A large principal-component correlation means the
    dominant direction of the representation is a purity direction, which is the
    confounding the gate was designed to test.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_independent_meta")     Excel metadata table of the CPTAC UCEC
        Independent cohort; columns Case_id and ABSOLUTE_tumor_purity are read.
    data("cptac_discovery_clinical")   clinical table of the CPTAC UCEC Discovery
        cohort; columns Proteomics_Participant_ID, Purity_Cancer and
        Proteomics_Tumor_Normal are read.
    work("harmonised_matrix_pattern")  harmonised per-cohort layer matrices
        (hcsv/{cohort}__{layer}.csv.gz); genes are rows, samples are columns. The
        cohort and layer tokens are param("cohort_tags") and param("layer_tags"),
        and the layers diagnosed are param("purity", "diagnostic_layers").

Outputs
    work("purity_diag_csv")    one row per cohort x layer pair.
    work("purity_diag_json")   the same table as JSON records.

Usage
    python 09_02_diagnose_purity_confounding.py
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
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.decomposition import PCA

from common.config import SEED, data, ensure_dir, param, work

# Column names of the two third-party purity sources. They are data-schema names
# taken verbatim from the files and are not tunable.
CASE_ID = "Case_id"
ABSOLUTE_PURITY = "ABSOLUTE_tumor_purity"
PARTICIPANT_ID = "Proteomics_Participant_ID"
PURITY_CANCER = "Purity_Cancer"
TUMOR_NORMAL = "Proteomics_Tumor_Normal"
TUMOR_LABEL = "Tumor"

# Cohorts whose purity can be read, as (config cohort key, label of the purity
# column) pairs. Only the two CPTAC UCEC cohorts publish a per-sample estimate;
# the TCGA cohort does not, so the purity-aware arms cannot be evaluated on it.
PURITY_SOURCES = (("independent", ABSOLUTE_PURITY), ("discovery", PURITY_CANCER))

# Analysis settings, all frozen in config/params.yaml.
GENE_COMPLETENESS = param("purity", "gene_completeness")
ABS_R_GENE = param("purity", "abs_r_gene")
MIN_SAMPLES = param("purity", "min_samples_diagnosed")
N_PC_DIAGNOSIS = param("purity", "n_pc_diagnosis")
ALPHA = param("statistics", "alpha")
# Layers diagnosed, in report order. The data source of every layer is its file
# name, and the matrix file is addressed by the layer's tag, so the tag is what
# the record carries: it is the key the diagnostics share with the other stages.
DIAGNOSTIC_LAYERS = [
    (name, param("layer_tags", name)) for name in param("purity", "diagnostic_layers")
]


def read_clinical_tsv(path: Path) -> pd.DataFrame:
    """Read a tab-separated clinical table of the CPTAC LinkedOmics export.

    Args:
        path: table to read.

    Returns:
        The table as read, with every column kept.
    """
    # See 09_01 for why both decoding options are needed.
    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def load_purity_independent(path: Path) -> pd.Series:
    """Build the sample -> purity series of the CPTAC UCEC Independent cohort.

    Args:
        path: Excel metadata table of that cohort.

    Returns:
        Purity indexed by Case_id, one row per case.
    """
    meta = pd.read_excel(path)
    frame = pd.DataFrame(
        {
            "id": meta[CASE_ID].astype(str),
            "purity": pd.to_numeric(meta[ABSOLUTE_PURITY], errors="coerce"),
        }
    ).dropna()
    # A purity of zero is a missing estimate in this cohort, not an observation.
    frame = frame[frame["purity"] > 0]
    # The metadata table repeats a case across its aliquots; keep one row per case.
    return frame.drop_duplicates("id").set_index("id")["purity"]


def load_purity_discovery(path: Path) -> pd.Series:
    """Build the sample -> purity series of the CPTAC UCEC Discovery cohort.

    Args:
        path: tab-separated clinical table of that cohort.

    Returns:
        Purity indexed by Proteomics_Participant_ID, one row per participant.
    """
    clinical = read_clinical_tsv(path)
    frame = pd.DataFrame(
        {
            "id": clinical[PARTICIPANT_ID].astype(str),
            "purity": pd.to_numeric(clinical[PURITY_CANCER], errors="coerce"),
            "tissue": clinical[TUMOR_NORMAL].astype(str),
        }
    )
    # Only the tumour aliquots carry a purity estimate; the normal ones have none.
    frame = frame[(frame["tissue"] == TUMOR_LABEL) & frame["purity"].notna()]
    return frame.drop_duplicates("id").set_index("id")["purity"]


def load_layer(cohort_tag: str, layer_tag: str) -> pd.DataFrame | None:
    """Load one harmonised cohort x layer matrix.

    Args:
        cohort_tag: cohort token as it appears in the file names, i.e. a value of
            param("cohort_tags").
        layer_tag: layer token as it appears in the file names, i.e. a value of
            param("layer_tags").

    Returns:
        The matrix with genes as rows and sample identifiers as columns, or None
        when the cohort has no matrix for that layer.
    """
    pattern = str(work("harmonised_matrix_pattern")).format(
        cohort=cohort_tag, layer=layer_tag
    )
    path = Path(pattern)
    if not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    # Sample identifiers arrive as strings in some files and as numbers in others;
    # forcing str makes every identifier comparison well defined.
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix


def diagnose_layer(
    cohort_key: str, layer_name: str, layer_tag: str, purity: pd.Series
) -> dict | None:
    """Measure the association between one layer matrix and sample purity.

    Args:
        cohort_key: config cohort key, used for the file name and the output row.
        layer_name: layer name from param("layers"), used to find the file name.
        layer_tag: layer token of the harmonised file names; it is what the
            output row records, because it is the key shared with other stages.
        purity: purity indexed by sample identifier.

    Returns:
        A diagnostics record, or None when the cohort has no matrix for that
        layer or too few samples carry a purity estimate.

    The recipe follows the confounding diagnostic of the gate: standardise each
    gene, correlate it with purity, and check whether purity also aligns with the
    leading principal components of the standardised matrix.
    """
    matrix = load_layer(param("cohort_tags", cohort_key), layer_tag)
    if matrix is None:
        return None
    shared = [sample for sample in matrix.columns if sample in purity.index]
    if len(shared) < MIN_SAMPLES:
        return None

    # Samples as rows, genes as columns.
    values = matrix[shared].T
    target = purity.reindex(shared).values.astype(float)
    # Genes that are mostly missing carry no usable correlation, and the median is
    # the imputation the rest of the pipeline uses for a sparse layer.
    values = values.loc[:, values.notna().mean() > GENE_COMPLETENESS].fillna(values.median())
    # z-score with ddof=1; a zero-variance gene would divide by zero, so those
    # genes are left at a unit scale (their correlation is undefined and the
    # pearsonr call below returns NaN for them, which is filtered out).
    standardised = (values - values.mean()) / values.std(ddof=1).replace(0, 1)

    # Gene-level correlation with purity.
    correlations = np.array(
        [
            pearsonr(standardised.iloc[:, j].values, target)[0]
            for j in range(standardised.shape[1])
        ]
    )
    correlations = correlations[np.isfinite(correlations)]
    fraction_high = float(np.mean(np.abs(correlations) > ABS_R_GENE))

    # Leading principal components of the same matrix, and their correlation with
    # purity. A purity-loaded PC means the transfer direction is a purity direction.
    n_components = min(
        N_PC_DIAGNOSIS, standardised.shape[0] - 1, standardised.shape[1]
    )
    pca = PCA(n_components=n_components, random_state=SEED).fit(standardised.values)
    scores = pca.transform(standardised.values)
    pc_correlations = [
        abs(pearsonr(scores[:, i], target)[0]) for i in range(n_components)
    ]
    # Mean squared gene-level correlation: the share of gene-level variance that
    # the linear purity term accounts for.
    mean_r2 = float(np.mean(correlations**2))

    if n_components > 1:
        print(
            f"  {cohort_key}/{layer_tag:12s} n={len(shared):3d} genes={standardised.shape[1]:5d} | "
            f"mean r2(purity)={mean_r2:.4f} | "
            f"genes with |r|>{ABS_R_GENE:g}: {fraction_high * 100:5.1f}% | "
            f"PC1|r|={pc_correlations[0]:.3f} PC2|r|={pc_correlations[1]:.3f}"
        )
    else:
        # Fewer than two components can be computed; the original report prints an
        # empty line here rather than a partial one.
        print("")

    return {
        "cohort": cohort_key,
        "layer": layer_tag,
        "n": len(shared),
        "n_gene": standardised.shape[1],
        "mean_r2_purity": round(mean_r2, 4),
        "frac_gene_absr_gt_0.3": round(fraction_high, 4),
        "max_pc_r": round(float(max(pc_correlations)), 4),
        "pc1_r": round(pc_correlations[0], 4),
        "pc2_r": round(pc_correlations[1], 4) if n_components > 1 else None,
    }


def main() -> None:
    """Diagnose purity confounding for every cohort x layer pair. Returns None."""
    purity_by_cohort = {
        "independent": load_purity_independent(data("cptac_independent_meta")),
        "discovery": load_purity_discovery(data("cptac_discovery_clinical")),
    }
    for cohort_key, column in PURITY_SOURCES:
        series = purity_by_cohort[cohort_key]
        print(
            f"Purity available: {cohort_key} {len(series)} samples ({column}) | "
            f"mean {series.mean():.4f} range [{series.min():.3f},{series.max():.3f}]"
        )

    # Purity itself is confounded with cohort: if the two distributions differ,
    # a transfer failure can be a purity difference rather than a platform
    # difference. The rank test is used because purity is not normal.
    _, p_value = mannwhitneyu(
        purity_by_cohort["independent"].values, purity_by_cohort["discovery"].values
    )
    verdict = "distributions differ" if p_value < ALPHA else "no significant difference"
    print(f"  purity distribution, independent vs discovery: Mann-Whitney p = {p_value:.3g} -> {verdict}")

    records = []
    for cohort_key in ("independent", "discovery"):
        for layer_name, layer_tag in DIAGNOSTIC_LAYERS:
            record = diagnose_layer(
                cohort_key, layer_name, layer_tag, purity_by_cohort[cohort_key]
            )
            if record is not None:
                records.append(record)

    table = pd.DataFrame(records)
    ensure_dir(work("purity_dir"))
    table.to_csv(work("purity_diag_csv"), index=False)
    print(f"\nwrote {work('purity_diag_csv')}")
    print(f"mean r2(purity) across layers: {table['mean_r2_purity'].mean():.4f} | "
          f"highest layer: {table.loc[table['mean_r2_purity'].idxmax(), 'layer']}")
    print(f"median max PC|r| across layers: {table['max_pc_r'].median():.4f}")
    with open(work("purity_diag_json"), "w", encoding="utf-8") as handle:
        json.dump(table.to_dict(orient="records"), handle, ensure_ascii=False, indent=1)
    print(f"wrote {work('purity_diag_json')}")


if __name__ == "__main__":
    main()

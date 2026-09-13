# Cross-cohort multi-omic integration benchmark (gynaecological tumours)

Replication package for a methodological benchmark asking **how well multi-omic
integration transfers across independent cohorts**, rather than within a single one.

> **Manuscript status.** The manuscript text is not included in this repository by
> design. This package contains the code, the harmonised intermediate objects, and
> the figure/table generation and verification harness.

---

## Headline results (all reproducible from this package)

| # | Result | Evidence strength |
|---|---|---|
| 1 | **Layer count has negative marginal value**: cross-cohort transfer decreases monotonically as layers are added | A — replicated in two fully independent toolchains |
| 2 | **"Best method" is not resolvable**: the winner flips in 50% of evaluation units; top-2 differ at p = 0.669; one method differs 17.6× between implementations | A |
| 3 | **The covariance layer is fragile**: per-gene corrections remove mean separation but not multivariate separability; first-order effect sizes transfer across platforms, second-order network structure retains ~35% | A — two orthogonal perturbation designs |
| 4 | Two pre-registered gates resolve as **undecidable**, not as failures | — |

---

## Repository layout

```
src/
  00_harmonisation/          layer harmonisation, ID mapping, numeric domains
  01_protein_bridge/         protein Δ(T−N) repair, biological anchor
  02_alignment_strategies/   ComBat / quantile / Δ-reference strategy matrix
  03_benchmark_python/       eight self-implemented Python method arms
  04_benchmark_R/            six official R packages (+ two reimplementations)
  05_pac_stability/          consensus-clustering stability (PAC)
  06_batch_testbed/          EEEC perfectly-crossed 2×2 batch design
  07_cross_platform/         LFQ vs TMT: first- vs second-order concordance
  08_purity_G2/              purity-aware arms and the G2 gate
  09_thresholds_G3/          threshold derivation and the G3 gate
  10_cross_implementation/   Python × R grid-level comparison
  11_figures/                figure generation (shared style module + per-figure scripts)
  12_tables/                 table extraction
  13_verification/           number verification and submission-compliance checks
  14_exports/                EPS / TIFF export
results/
  figures/                   all main and supplementary figures: PNG + PDF + EPS + TIF
  tables/                    12 machine-readable CSVs
docs/
  figure_index.md            figure → content mapping
  table_legends.md           table descriptions
environment/                 pinned Python and R package versions
```

---

## How to reproduce

The pipeline is staged and each stage reads only the previous stage's outputs.
Run from the repository root:

```bash
# Python 3.12 with environment/requirements.txt
python src/00_harmonisation/e1b_harmonize.py          # -> harmonised layers
python src/00_harmonisation/a5_concordance.py         # -> cross-cohort concordance
python src/01_protein_bridge/b1_bridge.py             # -> protein Δ repair
python src/02_alignment_strategies/e2b_ari.py         # -> strategy × layer ARI
python src/03_benchmark_python/f2_benchmark.py        # -> 8 Python arms
Rscript src/04_benchmark_R/r5_bench.R                 # -> 6 R packages
python src/05_pac_stability/h1_pac.py                 # -> PAC
python src/06_batch_testbed/k2_correct.py             # -> batch correction arms
python src/07_cross_platform/l2_platform2.py          # -> cross-platform ladder
python src/08_purity_G2/n6_pai.py                     # -> purity arms
python src/09_thresholds_G3/p4_g3_power.py            # -> G3 threshold derivation

# figures and tables
for f in src/11_figures/{r3_fig12,r7_fig3,r11_fig45,s1_figS123,s2_figS45,s3_figS6,s4_figS8,u4_figS7}.py; do
  python "$f"
done
python src/12_tables/t1_tables.py && python src/12_tables/t2_mdtables.py

# verification
python src/13_verification/c1_compliance.py           # submission compliance
python src/11_figures/r5_geomcheck.py                 # figure geometry harness
```

**Random seed** is 49 throughout. All figures are built at a fixed physical width so
that the smallest in-figure type stays ≥ 8 pt at print size; this is enforced
programmatically by `src/11_figures/figstyle.py::check`, which extracts the rendered
text bounding boxes and fails on sub-8 pt type, overlapping text, or content bleeding
outside the canvas.

---

## Data

All input data are previously published, de-identified public datasets; no ethical
approval was required. See [`DATA.md`](DATA.md) for sources, accessions and download
instructions. Raw inputs are **not** redistributed here — only the code needed to
fetch and harmonise them.

---

## What is deliberately not here

- **The manuscript text.** Not redistributed.
- **Raw third-party data.** Re-downloadable from the public sources listed in `DATA.md`.
- **Exploratory scripts.** Roughly 230 additional scripts were written during
  development; the 79 included here are the canonical pipeline that reproduces the
  reported results.

---

## Known limitations (disclosed in the manuscript)

- The reference partition used by the transfer metric is derived by K-means, which
  structurally favours K-means-family methods. The "simplest methods win" statement
  is scoped to that reference construction.
- `MOFA2` results are indicative: R and its basilisk-managed mofapy2 each ship a
  libomp, so `KMP_DUPLICATE_LIB_OK=TRUE` was required (the package warns this may
  affect results).
- Two packages (intNMF, MCIA) were delisted from CRAN and Bioconductor and are
  reimplemented from their original descriptions. They are labelled as equivalents
  and no substitute package was used in their place.

---

## Citation

If you use this code, please cite the manuscript (full reference to be added on
publication) and this archive via its Zenodo DOI.

## License

MIT — see [`LICENSE`](LICENSE).

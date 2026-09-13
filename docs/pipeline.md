# Pipeline

Execution order of the analysis. Each stage reads the artefacts of the stages
before it and writes its own into the work root. Stage directories are numbered
`01`–`15`; within a stage, file names are numbered `MM` so that sorting a
directory gives the order in which its scripts should run.

## Layout

```
config/            paths.yaml and params.yaml -- the only place a location or a
                   parameter is written down
src/common/        shared modules: config loader, plotting style
src/01_...15_.../  the analysis, in execution order
results/           published deliverables (figures, tables, verification)
docs/              this file, the coding standard, data sources, figure index
manuscript/        LaTeX sources of the manuscript and supplement
```

## Configuration

Nothing in `src/` contains a path or an analysis parameter. Both come from
`config/`:

- `config/paths.yaml` — every filesystem location, including the external data
  volume and the work root.
- `config/params.yaml` — every parameter, threshold and categorical vocabulary.

Any leaf can be redirected for a single run with an environment variable, using
the dotted path in upper case with a `GYN_` prefix:

```bash
# Run against a different data volume and work directory
GYN_DATA_ROOT=/mnt/omics GYN_WORK_ROOT=/scratch/gyn \
    python src/01_data_harmonisation/01_01_scan_layer_structure.py
```

The work root defaults to `<repository>/.work` and is not version controlled.

## Running a script

Every script is runnable from any directory and inserts `src/` on `sys.path`
itself, so `common.config` and `common.plotting_style` are importable:

```bash
python src/12_figures/12_02_build_figure_1_and_2.py
```

R scripts under `src/05_benchmark_r/` read the same two YAML files, so the R and
Python sides of the benchmark share one configuration.

## Stages

| Stage | Directory | What it does | Key outputs |
|---|---|---|---|
| 01 | `src/01_data_harmonisation` | Reads the raw TCGA/Xena and CPTAC matrices, reconciles identifier systems, maps every layer onto a common gene space, profiles numeric domains, measures per-layer cross-cohort concordance and resolves duplicate samples into one harmonised matrix per cohort × layer | `harmonised/`, `hcsv/`, `align_*.json` |
| 02 | `src/02_protein_bridge` | Builds the protein-layer bridge: raw TMT ratios are not comparable across cohorts, so the layer is re-expressed as tumour-minus-normal differences and validated against an independent mRNA–protein anchor | `bridge_*.json`, `b3_anchors.json` |
| 03 | `src/03_alignment_strategies` | Evaluates the five cross-cohort alignment strategies under gate G4 and aggregates their transfer ARIs | `ari_matrix.json`, `ari_agg.json` |
| 04 | `src/04_benchmark_python` | Runs the cross-cohort transfer benchmark for the eight self-implemented Python method arms over every layer subset and both evaluation domains, then aggregates and summarises | `bench_matrix.json`, `bench_summary.json` |
| 05 | `src/05_benchmark_r` | The same benchmark for the six R packages, using the identical protocol and the same configuration | `bench_r_matrix.csv` |
| 06 | `src/06_consensus_stability` | Builds consensus matrices by bootstrap resampling and computes PAC together with the degeneracy diagnostics that PAC alone would hide | `pac_matrix.json`, `pac_agg.json` |
| 07 | `src/07_batch_testbed` | Builds the EEEC batch testbed from the crossed 2×2 design, applies the nine correction arms and computes the five sign-fixed batch metrics | `eeec_batch/` |
| 08 | `src/08_cross_platform` | Cross-platform comparison between label-free LFQ and TMT: the comparability ladder, the effect-size stratified concordance and the correlation-network agreement | `platform/` |
| 09 | `src/09_purity_gate_g2` | Purity diagnostics and the six purity-aware arms evaluated against the frozen G2 baseline | `purity/`, `pai/`, `g2_refined.json` |
| 10 | `src/10_thresholds_gate_g3` | Derives the G2/G3 decision thresholds from the measured noise floor, measures the within-cohort reproducibility ceiling, and builds and guards the report documents | report PDFs |
| 11 | `src/11_cross_implementation` | Merges the Python and R toolchain results into one comparable table | merged benchmark table |
| 12 | `src/12_figures` | Builds every published figure and runs the geometric checker over the rendered output | `results/figures/` |
| 13 | `src/13_tables` | Builds every published table from the analysis artefacts and verifies the values against their source | `results/tables/` |
| 14 | `src/14_verification` | The audit layer: extracts every number from the manuscript sources and checks it against the artefact that produced it, verifies references and citations, and checks submission compliance | `results/verification/` |
| 15 | `src/15_exports` | Exports the figures in the formats the publisher requires, including native vector EPS and 300 dpi LZW TIFF | `results/figures/` |

## Dependency summary

```
01 harmonisation ──┬── 02 protein bridge ──┐
                   ├── 03 alignment ───────┤
                   └── 04 python benchmark ├── 11 cross-implementation ──┐
                       05 R benchmark ─────┘                            │
01 ── 06 stability ─────────────────────────────────────────────────────┤
01 ── 07 batch testbed ─────────────────────────────────────────────────┼── 12 figures ── 15 exports
01 ── 08 cross-platform ────────────────────────────────────────────────┤
02 ── 09 purity / G2 ───────────────────────────────────────────────────┤
01 ── 10 thresholds / G3 ───────────────────────────────────────────────┘
                                                                       └── 13 tables
                                                    all of the above ──┬── 14 verification
```

## Verification

The package is written so that its claims can be checked rather than trusted.
Three checks run over the outputs:

1. **Geometric** (`12_01_check_figure_geometry.py`) — extracts the bounding box
   of every rendered text element in each figure and fails on type below
   `output.min_font_pt` at print size, on pairwise text overlap, or on content
   leaving the canvas.
2. **Numeric** (`14_02`, `14_03`, `14_08`, `14_09`) — pulls every number out of
   the manuscript sources and looks it up in the analysis artefact that produced
   it, so a transcription error cannot survive a build.
3. **Compliance** (`14_10_check_submission_compliance.py`) — checks figure width
   against the journal limit, raster resolution, and the EPS vector audit
   (`14_11`) that distinguishes genuinely vector output from output that was
   rasterised by a converter.

## Reproducing the study

```bash
# 1. Point the configuration at your copy of the data (see docs/data_sources.md)
export GYN_DATA_ROOT=/path/to/data
export GYN_WORK_ROOT=/path/to/work

# 2. Run the stages in order
for d in src/0[1-9]_* src/1[0-5]_*; do
    for f in "$d"/*.py; do python "$f"; done
done

# 3. The R side reads the same configuration
Rscript src/05_benchmark_r/05_02_benchmark_r_packages.R
```

The seed is fixed globally at `params.yaml → seed`, so every stochastic step
reproduces exactly.

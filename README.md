# Cross-cohort multi-omic integration benchmark in gynaecological tumours

Replication package for a methodological benchmark asking **how well multi-omic
integration transfers across independent cohorts** — the evaluation target that
matters for any method that must generalise beyond the cohort it was fitted on.

Five public cohorts, two fully independent toolchains (eight self-implemented
Python methods over scikit-learn, six official R packages), and two orthogonal
perturbation designs.

## Findings

1. **Layer count has negative marginal value.** Cross-cohort transfer decreases
   monotonically as omics layers are added, in both evaluation domains and both
   toolchains. Two mechanisms operate together: intersecting layers collapses the
   analysable sample set (TCGA 539 → 390, CPTAC-UCEC Independent 138 → 84), and
   weakly comparable layers contribute cohort-specific variance that cannot be
   reproduced in the held-out cohort.

2. **The "best method" is not a resolvable claim.** The winning method changes in
   50% of evaluation units when the held-out cohort changes; the top two methods
   are statistically indistinguishable; and the same published method differs
   17.6-fold between two implementations of it.

3. **The covariance layer is the fragile layer.** In a perfectly crossed batch
   design, per-gene corrections drive mean separation to exactly chance while
   multivariate separability stays complete. Cross-platform, first-order effect
   sizes transfer at same-cohort level while second-order network structure
   retains about one third of that concordance.

Two of six pre-registered decision gates resolved as **undecidable** rather than
as failures — one for lack of information (the effect size that was pre-specified
requires 479 evaluation records where 22 exist), the other because the measuring
instrument was degenerate. The distinction, and a procedure for deriving decision
thresholds from the measured noise distribution instead of assuming them, is the
most transferable output of the work.

## Repository layout

```
config/       paths.yaml, params.yaml          every location and parameter
src/common/   config loader, plotting style    shared modules
src/01..15/   the analysis, in execution order
results/      figures, tables, verification    published deliverables
docs/         pipeline, coding standard, data sources, figure index
manuscript/   LaTeX sources of the manuscript and supplement
```

`docs/pipeline.md` describes each stage, its inputs and outputs, and the
dependency order. `docs/coding_standard.md` states the rules the code follows.

## Reproducing

```bash
export GYN_DATA_ROOT=/path/to/your/copy      # see docs/data_sources.md
export GYN_WORK_ROOT=/path/to/scratch        # defaults to ./.work

# Stages run in order; each directory's files are numbered in run order.
for d in src/0[1-9]_* src/1[0-5]_*; do
    for f in "$d"/*.py; do python "$f"; done
done

# The R side of the benchmark reads the same configuration.
Rscript src/05_benchmark_r/05_02_benchmark_r_packages.R
```

Nothing needs to be edited to relocate the analysis: every path and every
parameter is read from `config/`, and any single entry can be overridden with an
environment variable named after its dotted path (`GYN_DATA_ROOT`,
`GYN_WORK_ROOT`, `GYN_CLUSTERING_K_PRIMARY`, …). The random seed is fixed
globally at `params.yaml → seed`.

## Verification

Outputs are checked, not asserted:

- **Figure geometry** — every rendered text element is measured and the build
  fails on type below 8 pt at print size, on text overlap, or on content leaving
  the canvas.
- **Number tracing** — every number in the manuscript sources is extracted
  automatically and looked up in the artefact that produced it.
- **Submission compliance** — figure width against the journal's 180 mm limit,
  raster resolution, and a vector audit that distinguishes genuine vector EPS
  from output that a converter silently rasterised.

## Data

All inputs are previously published, de-identified public datasets. None are
redistributed; `docs/data_sources.md` lists every source and accession. No new
human or animal subjects were involved and no ethical approval was required.

## Licence

See `LICENSE`.

## Citation

See `CITATION.cff`.

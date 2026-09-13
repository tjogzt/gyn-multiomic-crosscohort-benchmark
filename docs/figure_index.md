# Figure index

All figures are provided in four formats: `.png` (preview, 300 dpi), `.pdf`
(vector), `.eps` (vector, production), `.tif` (300 dpi LZW, for journal systems).
All are ≤ 180 mm wide with in-figure type ≥ 8 pt at print size.

Every figure is reproducible from `config/` alone; the "script" column names the
file that writes it.

## Main figures

| File | Panels | Content | Reproducing script |
|---|---|---|---|
| `Fig1_availability_comparability` | a–d | Sample availability by layer; cross-cohort layer concordance; protein Δ(T−N) repair; mRNA–protein biological anchor | `src/12_figures/12_02_build_figure_1_and_2.py` |
| `Fig2_layer_negative_marginal` | a–d | Transfer ARI vs number of layers (both toolchains, both domains); sample-intersection collapse; layer-subset × method heatmap | `src/12_figures/12_02_build_figure_1_and_2.py` |
| `Fig3_method_not_resolvable` | a–e | Method ranking per toolchain; champion-flip count; cross-implementation agreement; PAC vs transfer | `src/12_figures/12_03_build_figure_3.py` |
| `Fig4_covariance_fragility` | a–e | Batch testbed; neighbourhood mixing; cross-platform ladder; effect-size dose–response; first- vs second-order agreement | `src/12_figures/12_04_build_figure_4_and_5.py` |
| `Fig5_thresholds_undecidable` | a–d | Noise-floor decomposition; required sample size; within-cohort reproducibility; degeneracy of the ceiling | `src/12_figures/12_04_build_figure_4_and_5.py` |

## Supplementary figures

| File | Content | Reproducing script |
|---|---|---|
| `FigS1_harmonisation_domains` | Identifier-system census; probe→gene mapping loss; numeric domains; duplicate-identifier audit | `src/12_figures/12_05_build_figures_s1_s2_s3.py` |
| `FigS2_protein_bridge` | Protein Δ across three cohorts; trimming control; mRNA–protein anchor; methylation degradation | `src/12_figures/12_05_build_figures_s1_s2_s3.py` |
| `FigS3_alignment_strategies` | Layer × strategy ARI heatmap; per-pair ΔARI; three exact identities; G4 verdict | `src/12_figures/12_05_build_figures_s1_s2_s3.py` |
| `FigS4_pac_stability` | Per-method PAC; **the PAC degeneracy trap**; PAC→ARI bins; scope of use | `src/12_figures/12_06_build_figures_s4_s5.py` |
| `FigS5_purity_arms` | Purity arms vs baseline; dose–response; win rate vs significance; variance explained | `src/12_figures/12_06_build_figures_s4_s5.py` |
| `FigS6_eeec_semantics` | Group-mean spectra; PC1 recovers the condition axis; within- vs across-group structure; variance explained | `src/12_figures/12_07_build_figure_s6.py` |
| `FigS7_availability_limits` | Layer counts and the 306-sample five-way intersection; leave-one-layer-out gain; analysis-set sizes; survival-data availability | `src/12_figures/12_09_build_figure_s7.py` |
| `FigS8_availability_landscape` | Data scale; per-cohort volume; analysis-ready sample sets; availability summary | `src/12_figures/12_08_build_figure_s8.py` |

---

## Colour semantics

Colour encodes **toolchain** and **verdict**, never decoration. The same meaning is
used in every figure. The palette is the traditional Chinese pigment set, named in
English throughout the code:

| Colour | Hex | Meaning |
|---|---|---|
| azurite | `#3D6BA8` | Python self-implemented toolchain |
| cinnabar | `#C23531` | R official-package toolchain |
| indigo | `#177CB0` | neutral / reference |
| rouge | `#9D2933` | failed / degenerate / undecidable |
| grey | `#999999` | auxiliary (raw values, no correction) |
| pine | `#3F6B3F` | passing / usable range |

Domain A and domain B are distinguished by **marker shape and line style**, so that
colour stays free to carry semantic meaning. Colours are defined once in
`src/common/plotting_style.py`; no figure script declares its own.

---

## Verification harness

`src/12_figures/12_01_check_figure_geometry.py` renders each figure and extracts the
bounding box of every text element, failing on:

1. any font below `output.min_font_pt` (8 pt) at print size,
2. pairwise text overlap (IoU > 0.12),
3. content extending beyond the canvas,
4. excessive canvas whitespace (reported, not fatal).

Note: panels created with `axis("off")` retain hidden tick labels, which must be
excluded from the geometry check or they produce false positives.

`src/14_verification/` additionally verifies that every number displayed in a
figure or table can be traced back to the analysis artefact that produced it.

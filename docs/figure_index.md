# Figure index

All figures are provided in four formats: `.png` (preview, 300 dpi), `.pdf` (vector),
`.eps` (vector, production), `.tif` (300 dpi LZW, journal systems).
All are ≤ 180 mm wide with in-figure type ≥ 8 pt at print size.

## Main figures

| File | Panels | Content | Reproducing script |
|---|---|---|---|
| `Fig1_availability_comparability` | a–d | Sample availability by layer; cross-cohort layer concordance; protein Δ(T−N) repair; mRNA–protein biological anchor | `src/11_figures/r3_fig12.py` |
| `Fig2_layer_negative_marginal` | a–d | Transfer ARI vs number of layers (both toolchains, both domains); sample intersection collapse; layer-subset × method heatmap | `src/11_figures/r3_fig12.py` |
| `Fig3_method_not_resolvable` | a–e | Method ranking per toolchain; champion-flip count; cross-implementation agreement; PAC vs transfer | `src/11_figures/r7_fig3.py` |
| `Fig4_covariance_fragility` | a–e | Batch testbed; neighbourhood mixing; cross-platform ladder; effect-size dose–response; first- vs second-order | `src/11_figures/r11_fig45.py` |
| `Fig5_thresholds_undecidable` | a–d | Noise-floor decomposition; required sample size; within-cohort reproducibility; degeneracy of the ceiling | `src/11_figures/r11_fig45.py` |

## Supplementary figures

| File | Content | Reproducing script |
|---|---|---|
| `FigS1_harmonisation_domains` | ID-system census; probe→gene mapping loss; numeric domains; duplicate-ID audit | `src/11_figures/s1_figS123.py` |
| `FigS2_protein_bridge` | Protein Δ across three cohorts; trimming control; mRNA–protein anchor; methylation degradation | `src/11_figures/s1_figS123.py` |
| `FigS3_alignment_strategies` | Layer × strategy ARI heatmap; per-pair ΔARI; three exact identities; G4 verdict | `src/11_figures/s1_figS123.py` |
| `FigS4_pac_stability` | Per-method PAC; **the PAC degeneracy trap**; PAC→ARI bins; scope of use | `src/11_figures/s2_figS45.py` |
| `FigS5_purity_arms` | Purity arms vs baseline; dose–response; win rate vs significance; variance explained | `src/11_figures/s2_figS45.py` |
| `FigS6_eeec_semantics` | Group-mean spectra; PC1 recovers the condition axis; within- vs across-group structure; variance explained | `src/11_figures/s3_figS6.py` |
| `FigS7_availability_limits` | Layer counts and the 306-sample five-way intersection; leave-one-layer-out gain; analysis-set sizes; survival-data availability | `src/11_figures/u4_figS7.py` |
| `FigS8_availability_landscape` | Data scale; per-cohort volume; analysis-ready sample sets; availability summary | `src/11_figures/s4_figS8.py` |

---

## Colour semantics

Colour encodes **toolchain** and **verdict**, never decoration. The same meaning is
used in every figure:

| Colour | Hex | Meaning |
|---|---|---|
| 石青 (azurite) | `#3D6BA8` | Python self-implemented toolchain |
| 朱砂 (cinnabar) | `#C23531` | R official-package toolchain |
| 靛青 (indigo) | `#177CB0` | neutral / reference |
| 胭脂 (rouge) | `#9D2933` | failed / degenerate / undecidable |
| grey | `#999999` | auxiliary (raw values, no correction) |
| pine | `#3F6B3F` | passing / usable range |

Domain A and Domain B are distinguished by **marker shape and line style**, so that
colour stays free to carry semantic meaning.

---

## Verification harness

`src/11_figures/figstyle.py::check` renders each figure and extracts the bounding box
of every text element, failing on:

1. any font below 8 pt,
2. pairwise text overlap (IoU > 0.12),
3. content extending beyond the canvas,
4. excessive canvas whitespace (reported).

Note: panels created with `axis("off")` retain hidden tick labels, which must be
excluded from the geometry check or they produce false positives.

`src/13_verification/` additionally verifies that every number displayed in a figure
or table can be traced back to an analysis artefact.

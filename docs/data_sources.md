# Data sources

All input data are previously published, de-identified public datasets. No new
human or animal subjects were involved and no ethical approval was required. Raw
inputs are **not** redistributed in this repository; download them from the
sources below and point `config/paths.yaml → data_root` at your copy, or set the
`GYN_DATA_ROOT` environment variable.

The locations of every input named here are declared in `config/paths.yaml`; no
script contains a literal path.

---

## Cohorts

| Code | Cohort | Role | Source |
|---|---|---|---|
| **TCGA** | TCGA Uterine Corpus Endometrial Carcinoma | Reference cohort (domain B) | UCSC Xena / GDC |
| **Discovery** | CPTAC-UCEC Discovery | External platform cohort (domain B) | LinkedOmics / PDC |
| **Independent** | CPTAC-UCEC Independent (Confirmatory) | External platform cohort (domain B) | LinkedOmics |
| **OV** | CPTAC-OV | Cohort of the second evaluation domain (domain A) | LinkedOmics |
| **EEEC** | Early-onset endometrioid EC proteogenomics | Batch testbed and cross-platform reference | PRIDE **PXD046507** |

The sample-ID overlap between TCGA-UCEC and each CPTAC cohort is **zero**,
verified by direct set intersection. Cross-cohort transfer is therefore genuine
external validation, not a re-split of one cohort.

---

## Layer files used

### TCGA-UCEC (via UCSC Xena)

| Layer | File |
|---|---|
| mRNA | `tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz` |
| miRNA | `UCEC/TCGA.UCEC.sampleMap_miRNA_HiSeq_gene.gz` |
| Methylation | `UCEC/TCGA.UCEC.sampleMap_HumanMethylation450.gz` |
| RPPA | `UCEC/TCGA.UCEC.sampleMap_RPPA.gz` |
| CNA | `UCEC/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz` |
| Clinical | `UCEC/TCGA.UCEC.sampleMap_UCEC_clinicalMatrix` |
| Survival | `UCEC/survival_UCEC_survival.txt` |
| Mutations | `UCEC/mc3_UCEC_mc3.txt.gz` |

> **Important.** The pan-cancer `EB++AdjustPANCAN_...` matrix contains samples
> from **all** TCGA cohorts. It must be restricted to the UCEC sample universe
> before use; all layers are intersected with the UCEC primary-tumour sample
> list. Skipping this step inflates the mRNA sample count from 532 to 9,702.

> **Also note.** The UCEC clinical matrix does **not** carry the molecular
> subtype annotation (POLE / MSI / CN-low / CN-high). Only pan-cancer MSI and CNA
> columns are present, so subtype strata cannot be reconstructed reliably from it.
> Any subtype-level panel must be derived from the TCGA marker paper instead, and
> the derivation must be reported.

### CPTAC (via LinkedOmics)

Per-cohort collections: `CPTAC_UCEC_Discovery_LinkedOmics`,
`CPTAC_UCEC_independent_LinkedOmics`, `CPTAC_OV_Prospective_LinkedOmics`.
Layers used: RNA-seq (gene level), proteome (gene level, median-polished log2
tumour/normal ratio), methylation, CNA (gene level, log2).

### EEEC (via PRIDE PXD046507)

MaxQuant 1.6.15 / timsTOF / TIMS-DDA / LFQ / FDR 1%. The archive is 684.8 GB
across 277 RAW files; the four quantitative and metadata files needed here were
retrieved by **remote zip64 central-directory parsing with HTTP range requests** —
68.6 MB compressed, 0.010% of the archive. No full download is performed.

Sample naming encodes a **perfectly crossed 2×2 design**: batch (`E`, `L`) ×
condition (`can`, `norm`), 49 samples per cell, 196 in total. Group-level
semantics were decoded from the data itself. A per-sample phenotype key is not
publicly obtainable — four sources were exhausted: the PRIDE archive manifest
(188,591 members), five supplementary files, the linked GitHub repository (110
files), and all 522 sample names in the GSA accession `HRA003319`.

---

## Harmonisation

All layers are mapped to **gene symbol** space:

- TCGA mRNA via `tcga_RSEM_Hugo_norm_count`
- TCGA 450K via the GDC probe map `illuminaMethyl450_hg19_GPL16304_TCGAlegacy`,
  which maps 344,298 of 485,577 probes (70.9%)

Identifier systems differ irreconcilably across sources — TCGA mRNA mixes 29
Entrez numeric identifiers into a symbol-like space, RPPA carries antibody names
(245 entries, whose intersection with the CPTAC gene-level protein matrix is
empty), and the 450K array carries probe identifiers.

The **five-way intersection** (mRNA, miRNA, methylation, RPPA, CNA) is 306
primary tumours. Samples are intersected **per layer subset**, and the resulting
count is reported for every subset rather than only for the full intersection.
No layer is ever imputed or broadcast; these are the only honest alternatives,
and both would introduce constructed data.

Leave-one-layer-out gains are highly uneven: dropping RPPA raises the analysable
set from 306 to 388 (+82) and dropping miRNA to 326 (+20), whereas dropping mRNA,
CNA or methylation changes it by only 1–3 samples.

---

## Derived objects

The pipeline writes harmonised matrices and stage artefacts into the **work
root**, which is declared in `config/paths.yaml → work_root` and defaults to
`<repository>/.work`. It is reproducible from the sources above and is excluded
from version control.

Set `GYN_WORK_ROOT` to keep the derived objects somewhere else:

```bash
GYN_WORK_ROOT=/scratch/gyn-work python src/01_data_harmonisation/01_08_harmonise_all_layers.py
```

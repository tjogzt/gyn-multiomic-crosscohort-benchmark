# Data sources

All input data are previously published, de-identified public datasets. No new human
or animal subjects were involved and no ethical approval was required. Raw inputs are
not redistributed in this repository; download them from the sources below.

---

## Cohorts

| Code | Cohort | Role | Source |
|---|---|---|---|
| **TCGA-UCEC** | TCGA Uterine Corpus Endometrial Carcinoma | Reference cohort (Domain B) | UCSC Xena / GDC |
| **V2 / `dis`** | CPTAC-UCEC Discovery | External platform cohort (Domain B) | LinkedOmics / PDC |
| **V1 / `ind`** | CPTAC-UCEC Independent (Confirmatory) | External platform cohort (Domain B) | LinkedOmics |
| **V3 / `OV`** | CPTAC-OV | Third cohort (Domain A) | LinkedOmics |
| **EEEC** | Early-onset endometrioid EC proteogenomics | Batch testbed + cross-platform | PRIDE **PXD046507** |

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

> **Important.** The pan-cancer `EB++AdjustPANCAN_...` matrix contains samples from
> **all** TCGA cohorts. It must be restricted to the UCEC sample universe before use —
> all layers are intersected with the UCEC primary-tumour sample list. Skipping this
> step inflates the mRNA sample count from 532 to 9,702.

### CPTAC (via LinkedOmics)

Per-cohort collections: `CPTAC-UCEC-independent`, `CPTAC-UCEC-discovery`, `CPTAC-OV`.
Layers used: RNA-seq (gene level), proteome (gene level, median-polished log2
tumour/normal ratio), methylation, CNA (gene level, log2).

### EEEC (via PRIDE PXD046507)

MaxQuant 1.6.15 / timsTOF / TIMS-DDA / LFQ / FDR 1%. The archive is 684.8 GB
(277 RAW files); the four quantitative and metadata files needed here were extracted
by **remote zip64 central-directory parsing with HTTP Range requests** — 68.6 MB
compressed, 0.010% of the original. The extraction scripts are in
`src/06_batch_testbed/`.

Sample naming encodes a **perfectly crossed 2×2 design**: batch (`E`, `L`) × condition
(`can`, `norm`), 49 samples per cell, 196 total. Group-level semantics were decoded
from the data itself; a per-sample phenotype key was not publicly obtainable (four
sources exhausted: the PRIDE archive manifest, five supplementary files, the linked
GitHub repository, and the GSA accession listing).

---

## Harmonisation

All layers are mapped to **gene symbol** space:

- TCGA mRNA via `tcga_RSEM_Hugo_norm_count`
- TCGA 450K via the GDC probe map `illuminaMethyl450_hg19_GPL16304_TCGAlegacy`
  (344,298 / 485,577 probes = 70.9% mapped)

The **five-way intersection** (mRNA, miRNA, methylation, RPPA, CNA) is 306 primary
tumours. Samples are intersected **per layer subset**, never imputed or broadcast.

---

## Sample de-duplication

The sample-ID overlap between TCGA-UCEC and each CPTAC cohort is **zero** (verified
directly). Cross-cohort transfer is therefore genuine external validation, not a
re-split of one cohort.

---

## Derived objects

The pipeline writes harmonised matrices to a working directory
(`$WORKDIR`, default `/tmp/gyn_retyping`). These are derived artefacts, reproducible
from the sources above; they are not committed here.

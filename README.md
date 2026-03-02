# Multimodal Sequence Embeddings for Genetic Interaction Prediction

**Adita Sivakumar, Arjun Gurjar, Marmik Chaudari**

---

## Overview

Gene interactions (GIs) encode epistatic regulatory dependencies, context-dependent cellular responses, and complex disease risk. Due to the combinatorial complexity of GIs, comprehensive datasets remain experimentally intractable, making generalizable models essential across existing fitness and transcriptome-resolved datasets.

Deep learning models like GEARS report improvements on unseen gene pairs, but independent benchmarks show that models do not consistently outperform simple additive baselines. We argue that robust GI prediction requires representing each layer of biological organization — **genome, epigenome, transcriptome, and proteome** — as a multimodal embedding that captures the regulatory grammar underlying epistasis.

---

## Proposal

For each gene, we construct multimodal representations from five embedding types, then combine gene embeddings via a learned interaction layer to predict the **non-additive** component of genetic interactions. We predict and evaluate only the non-additive residual (not total expression) to avoid conflating epistatic signal with systematic transcriptome-wide shifts that inflate standard metrics.

### Embedding Modalities

| Modality | Models | Input | Output |
|---|---|---|---|
| **Genomic (gLM)** | Evo2, Nucleotide Transformer | DNA regulatory sequence (±2 kb around TSS) | Sequence-level embedding |
| **Epigenomic (eFM)** | Enformer, Borzoi, AlphaGenome | 196,608 bp TSS-proximal sequence | Cis-regulatory context embedding |
| **Single-cell (scFM)** | Stack, scPRINT2 | Cell-state transcriptomic context | Cell-state embedding |
| **Protein (pLM)** | ESM C, ESM2 | Protein amino acid sequence | Mean-residue protein function embedding |
| **Protein interactions** | STRING | Network graph | Interaction network embedding |

### Prediction Targets

For each gene pair (A, B):

- **(a) Fitness epsilon score** — non-additive component of the double-perturbation fitness effect (Horlbeck et al.)
- **(b) Transcriptomic residual** — `expression(AB) − additive_expectation(A, B)` from single-perturbation baselines (Norman et al.)

---

## Experiments

### 1. Embedding Quality Assessment
Extract embeddings for all genes in Horlbeck et al., cluster using K-means, and reduce dimensionality with PCA/UMAP. Assess whether genes cluster by known functional identity, pathway annotations, and paralog status. This validates that embeddings recover regulatory and functional gene groupings relevant to GI biology.

### 2. Fitness Benchmark (Horlbeck et al.)
Train models predicting pairwise fitness scores under 2/2-unseen gene-level holdout, ablating one modality at a time to identify the most informative modalities.
- **Primary metrics:** AUROC for GI sign; Pearson *r* for effect size
- **GI categories:** synthetic lethality, synthetic sickness, buffering, suppression
- **Evaluation:** precision@10 per subtype, baseline-corrected against the additive null

### 3. Zero-Shot Transcriptome Prediction (Norman et al.)
Train on each embedding modality and multimodal combinations; evaluate non-additive transcriptomic residuals under 2/2-unseen splits.
- **Primary metrics:** Systema-framework Pearson correlation; GI subtype precision@10
- **Baselines:** GEARS; additive expectation from single perturbations

### 4. Low-Annotation Gene Generalization
Stratify Horlbeck genes by STRING edge density and GO annotation completeness. Test whether sequence-derived embeddings (gLM, eFM, pLM) achieve comparable performance for poorly annotated genes relative to well-annotated ones.

### 5. Cross-Cell-Type Generalization
Train on K562 and evaluate on Jurkat using Horlbeck fitness scores. Assess whether conditioning on eFM and scFM embeddings improves zero-shot transfer. Fine-tuning scFMs/eFMs on scRNA/ATAC-seq data and leveraging Stack's in-context learning can enable few-shot learning of cell-type-specific regulatory grammar.

---

## Repository Structure

```
aixbio/
├── src/
│   ├── data/
│   │   ├── download_horlbeck.py        # Download Horlbeck et al. 2018 fitness GI data
│   │   ├── download_norman.py          # Download Norman et al. 2019 transcriptome GI data
│   │   ├── get_tss.py                  # Extract TSS windows from GENCODE/hg38
│   │   └── build_dataset.py            # Build pairwise feature matrices and gene splits
│   ├── embeddings/
│   │   ├── genomic/
│   │   │   ├── evo2.py                 # Evo2 (NVIDIA BioNeMo API)
│   │   │   └── nucleotide_transformer.py  # Nucleotide Transformer
│   │   ├── epigenomic/
│   │   │   ├── enformer.py             # Enformer (K562 DNASE, H3K27ac)
│   │   │   ├── borzoi.py               # Borzoi
│   │   │   └── alphagenome.py          # AlphaGenome
│   │   ├── singlecell/
│   │   │   ├── stack.py                # Stack scFM
│   │   │   └── scprint2.py             # scPRINT2
│   │   ├── protein/
│   │   │   ├── esm_c.py                # ESM C (EvolutionaryScale)
│   │   │   └── esm2.py                 # ESM2
│   │   └── string_db.py                # STRING protein interaction embeddings
│   ├── models/
│   │   ├── interaction_layer.py        # Learned gene-pair interaction layer
│   │   ├── go_baseline.py              # GO-term binary feature baseline
│   │   └── train_evaluate.py           # Training loop and evaluation
│   └── eval/
│       └── metrics.py                  # AUROC, Pearson r, precision@k, Systema metrics
├── experiments/
│   ├── 01_embedding_quality.py         # Experiment 1: embedding clustering/UMAP
│   ├── 02_fitness_benchmark.py         # Experiment 2: Horlbeck fitness prediction
│   ├── 03_transcriptome_zeroshot.py    # Experiment 3: Norman transcriptome prediction
│   ├── 04_low_annotation_genes.py      # Experiment 4: low-annotation generalization
│   └── 05_cross_celltype.py            # Experiment 5: K562 → Jurkat transfer
├── data/
│   ├── horlbeck/                       # Horlbeck et al. supplementary tables
│   ├── norman/                         # Norman et al. perturbation data
│   └── annotations/                    # GENCODE GTF, GO annotations
└── results/                            # Experiment outputs, figures, evaluation CSVs
```

---

## Setup

> **Note:** Full environment setup instructions will be added as models are integrated.

```bash
git clone https://github.com/agurjar123/aixbio.git
cd aixbio
pip install -r requirements.txt
```

API keys required (add to `.env`):
```
EVO2_API_KEY=your_nvidia_bionemo_key
```

---

## Running the Pipeline

### Data preparation
```bash
python src/data/download_horlbeck.py   # Downloads fitness GI pairs → data/horlbeck/
python src/data/get_tss.py             # Extracts TSS windows from GENCODE hg38
python src/data/build_dataset.py       # Builds train/test splits under gene-level holdout
```

### Embedding extraction
```bash
python src/embeddings/epigenomic/enformer.py        # Enformer epigenomic embeddings
python src/embeddings/genomic/evo2.py               # Evo2 genomic embeddings
# Additional modalities: nucleotide_transformer, borzoi, alphagenome, esm_c, esm2, string_db
```

### Experiments
```bash
python experiments/01_embedding_quality.py
python experiments/02_fitness_benchmark.py
python experiments/03_transcriptome_zeroshot.py
python experiments/04_low_annotation_genes.py
python experiments/05_cross_celltype.py
```

---

## References

1. Costanzo et al. *Science* 2016
2. Costanzo et al. *Science* 2021 (GxGxE environmental robustness)
3. Horlbeck et al. *Cell* 2018
4. Norman et al. *Science* 2019
5. Roohani et al. *Nat Biotechnol* 2024 (GEARS)
6. Ahlmann-Eltze et al. *Nat Methods* 2025
7. Viñas Torné et al. *Nat Biotechnol* 2025 (Systema)
8. Wu et al. *NeurIPS* 2025
9. Brixi et al. *bioRxiv* 2025 (Evo2)
10. Dalla-Torre et al. *Nat Methods* 2025 (Nucleotide Transformer)
11. Avsec et al. *Nat Methods* 2021 (Enformer)
12. Linder et al. *Nat Genet* 2025 (Borzoi)
13. Avsec et al. *Nature* 2026 (AlphaGenome)
14. Dong et al. *bioRxiv* 2026 (Stack); Adduri et al. *bioRxiv* 2025 (State)
15. Kalfon et al. *bioRxiv* 2025 (scPRINT2)
16. ESM Team, EvolutionaryScale 2024 (ESM C)
17. Lin et al. *Science* 2023 (ESM2)
18. Szklarczyk et al. *NAR* 2025 (STRING)
19. Cole et al. *bioRxiv* 2026
20. Replogle et al. *Cell* 2022

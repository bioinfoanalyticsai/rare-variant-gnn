# 🧬 Rare Variant Pathogenicity Prediction — GNN Pipeline

End-to-end **Nextflow** pipeline for predicting the pathogenicity of rare genetic variants using **Graph Attention Networks (GAT)**. The model leverages protein-protein interaction (PPI) networks, multi-scale sequence conservation, and functional annotations to classify variants as **pathogenic** or **benign**.

---

## 📐 Architecture Overview

```
VCF Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                                 │
│  Parse VCF → Filter rare (AF < 0.001) → VEP annotate → Merge labels│
│  → Stratified gene-level train/val/test split                       │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE EXTRACTION  (parallel)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Sequence     │  │ Conservation │  │ Gene-level               │  │
│  │ CADD,SIFT,   │  │ PhyloP,      │  │ pLI, LOEUF, GTEx        │  │
│  │ PolyPhen,    │  │ PhastCons,   │  │ expression, OMIM         │  │
│  │ REVEL, BLOSUM│  │ GERP++       │  │                          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         └─────────────────┴──────────────────────┘                 │
│                           │ Combined feature matrix                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GRAPH CONSTRUCTION                               │
│  STRING PPI network → Gene-level graph                              │
│  → Assign variant features to nodes                                 │
│  → Normalize → Serialize to PyTorch Geometric (.pt)                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GNN TRAINING                                   │
│                                                                     │
│  ┌──────────────────────────────────────────────┐                  │
│  │    Graph Attention Network (GATv2)            │                  │
│  │                                              │                  │
│  │  node features (64-dim)                      │                  │
│  │       │                                      │                  │
│  │  FeatureEncoder (MLP)                        │                  │
│  │       │                                      │                  │
│  │  GATv2Block × L layers    ← PPI edges        │                  │
│  │  (multi-head attention,                      │                  │
│  │   residual connections,                      │                  │
│  │   edge dropout)                              │                  │
│  │       │                                      │                  │
│  │  JumpingKnowledge aggregation                │                  │
│  │  (cat / max / LSTM)                          │                  │
│  │       │                                      │                  │
│  │  MLP Classifier                              │                  │
│  │       │                                      │                  │
│  │  Pathogenicity Score ∈ [0, 1]               │                  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  Loss: Focal Loss (α=0.25, γ=2.0)  ← handles class imbalance      │
│  Optimizer: AdamW + CosineAnnealing                                 │
│  Early stopping on val AUROC                                        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION                                      │
│  AUROC, AUPRC, F1, Sensitivity, Specificity, PPV, NPV              │
│  ROC/PR curves, calibration plots, HTML report                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rare-variant-gnn/
├── main.nf                         # Main Nextflow pipeline
├── nextflow.config                 # Configuration (Docker/SLURM/AWS)
├── requirements.txt
├── conf/
│   └── environment.yml             # Conda environment
├── workflows/
│   ├── preprocessing.nf            # VCF parsing, filtering, labeling
│   ├── feature_extraction.nf       # Feature computation
│   ├── graph_construction.nf       # PPI graph building
│   ├── gnn_training.nf             # Model training
│   ├── gnn_prediction.nf           # Inference
│   └── evaluation.nf               # Metrics and plots
├── modules/
│   ├── parse_vcf.nf
│   ├── filter_variants.nf
│   ├── annotate_variants.nf
│   ├── merge_labels.nf
│   ├── split_dataset.nf
│   ├── extract_sequence_features.nf
│   ├── extract_conservation.nf
│   ├── extract_structural_features.nf
│   ├── extract_gene_features.nf
│   ├── combine_features.nf
│   ├── build_protein_graph.nf
│   ├── assign_node_features.nf
│   ├── serialize_graph.nf
│   ├── train_gnn.nf
│   ├── predict_gnn.nf
│   ├── evaluate_model.nf
│   └── hyperparameter_tune.nf
├── models/
│   └── gnn_model.py                # RareVariantGNN (GATv2 + JK + FocalLoss)
├── bin/
│   ├── parse_vcf.py
│   ├── filter_variants.py
│   ├── merge_labels.py
│   ├── split_dataset.py
│   ├── extract_sequence_features.py
│   ├── extract_conservation.py
│   ├── combine_features.py
│   ├── build_protein_graph.py
│   ├── assign_node_features.py
│   ├── serialize_graph.py
│   ├── train_gnn.py
│   ├── predict_gnn.py
│   ├── evaluate_model.py
│   └── hyperparameter_tune.py
├── scripts/
│   └── generate_test_data.py       # Synthetic data for testing
└── data/
    └── example/                    # Auto-generated test data
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Clone
git clone https://github.com/bioinfoanalyticsai/rare-variant-gnn
cd rare-variant-gnn

# Create conda environment
conda env create -f conf/environment.yml
conda activate rare-variant-gnn

# Or pip
pip install -r requirements.txt
```

### 2. Generate Test Data

```bash
# Default (500 variants, data/example/ output)
python scripts/generate_test_data.py
# Creates: data/example/{test_variants.vcf, clinvar_variants.tsv, string_ppi.tsv, ...}

# Custom size and location
python scripts/generate_test_data.py \
    --outdir /my/data \
    --n-variants 2000 \
    --n-labeled 600 \
    --seed 123
```

### 3. Run the Full Pipeline

```bash
# Full pipeline (train + predict + evaluate)
nextflow run main.nf \
    --input_vcf data/example/test_variants.vcf \
    --known_variants data/example/clinvar_variants.tsv \
    --ppi_network data/example/string_ppi.tsv \
    --gene_annot data/example/gene_annotations.tsv \
    --conservation data/example/conservation_scores.tsv \
    --outdir results/ \
    --gnn_epochs 100 \
    --gnn_hidden 128 \
    --gnn_layers 3 \
    -profile docker
```

### 4. HPC (SLURM)

```bash
nextflow run main.nf \
    --input_vcf variants.vcf.gz \
    --known_variants clinvar.tsv \
    --outdir results/ \
    -profile slurm \
    -bg
```

### 5. Prediction Only (Pretrained Model)

```bash
nextflow run main.nf \
    --mode predict \
    --input_vcf new_samples.vcf.gz \
    --pretrained_model results/models/best_model.pt \
    --outdir predictions/
```

### 6. Hyperparameter Optimization

```bash
# Direct Python (before full pipeline)
python bin/hyperparameter_tune.py \
    --graph results/graph/pyg_graph.pt \
    --labels results/preprocessing/splits/split_labels.tsv \
    --trials 100 \
    --output best_hparams.json
```

---

## 🔬 Model Details

### Graph Attention Network (GATv2)

| Component              | Details                                          |
|------------------------|--------------------------------------------------|
| Input features         | 64-dim (sequence + conservation + gene features) |
| Hidden dimension       | 128                                              |
| GATv2 layers           | 3                                                |
| Attention heads        | 4 per layer                                      |
| JumpingKnowledge       | Concatenation mode                               |
| Dropout                | 0.3 (node) + 0.1 (edge)                         |
| Loss                   | Focal Loss (α=0.25, γ=2.0)                      |
| Optimizer              | AdamW (lr=1e-3, weight_decay=1e-4)              |
| Scheduler              | CosineAnnealingWarmRestarts                      |

### Input Features

| Category      | Features                                                                     |
|---------------|------------------------------------------------------------------------------|
| Sequence      | CADD, SIFT, PolyPhen2, REVEL, M-CAP, BLOSUM62, AA physicochemical changes   |
| Conservation  | PhyloP100, PhastCons100, GERP++, SiPhy                                       |
| Structural    | Protein domain overlap, PTM sites, secondary structure                       |
| Gene-level    | pLI, LOEUF, missense-z, CDS length, GTEx expression, OMIM disease status     |

### Graph Structure

- **Nodes**: Genes (with variant-level features aggregated by max pathogenicity score)
- **Edges**: STRING PPI interactions (score ≥ 400 by default)
- **Self-loops**: Added to all nodes
- **k-hop neighborhood**: 2 hops

---

## 📊 Expected Performance

On ClinVar-labeled datasets (trained on pathogenic/likely-pathogenic vs benign/likely-benign):

| Metric       | Typical Range |
|--------------|---------------|
| AUROC        | 0.88 – 0.95   |
| AUPRC        | 0.75 – 0.88   |
| Sensitivity  | 0.82 – 0.91   |
| Specificity  | 0.85 – 0.94   |
| F1 (optimal) | 0.79 – 0.88   |

---

## 🛠️ Configuration Options

| Parameter           | Default   | Description                              |
|---------------------|-----------|------------------------------------------|
| `--af_threshold`    | 0.001     | Max allele frequency (rare = AF < 0.1%)  |
| `--graph_k_hops`    | 2         | Neighborhood size for message passing    |
| `--gnn_epochs`      | 100       | Training epochs                          |
| `--gnn_lr`          | 0.001     | Learning rate                            |
| `--gnn_hidden`      | 128       | Hidden layer dimension                   |
| `--gnn_layers`      | 3         | Number of GAT layers                     |
| `--batch_size`      | 64        | Training batch size                      |
| `--genome_build`    | GRCh38    | Reference genome                         |

---

## 📄 Output Files

```
results/
├── preprocessing/
│   ├── vcf_parsed/parsed_variants.tsv
│   ├── filtered/filtered_variants.tsv
│   ├── annotated/annotated_variants.tsv
│   └── splits/split_labels.tsv
├── features/
│   └── combined/node_features.tsv
├── graph/
│   ├── pyg_graph.pt              ← PyTorch Geometric graph
│   └── graph_stats.json
├── models/
│   ├── best_model.pt             ← Best checkpoint
│   ├── training_logs.json
│   └── checkpoints/
├── predictions/
│   ├── predictions.tsv           ← Binary predictions
│   ├── probabilities.tsv         ← Pathogenicity scores
│   └── variant_scores.tsv        ← Ranked variant list
├── evaluation/
│   ├── metrics.json
│   ├── eval_report.html          ← Interactive HTML report
│   └── plots/
│       ├── roc_pr_curves.png
│       ├── score_distribution.png
│       └── confusion_matrix.png
└── pipeline_info/
    ├── report.html               ← Nextflow execution report
    ├── timeline.html
    └── trace.txt
```

---

## 📚 References

- Velickovic et al. (2018) — [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- Brody et al. (2022) — [How Attentive are Graph Attention Networks? (GATv2)](https://arxiv.org/abs/2105.14491)
- Xu et al. (2018) — [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536)
- Lin et al. (2017) — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Szklarczyk et al. (2023) — [STRING v12: protein-protein association networks](https://doi.org/10.1093/nar/gkac1000)
- Landrum et al. (2018) — [ClinVar: improving access to variant interpretations](https://doi.org/10.1093/nar/gkx1153)

---

## 📜 License

MIT License — see LICENSE file.

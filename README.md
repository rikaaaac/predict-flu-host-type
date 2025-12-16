# ESM-2 Host Prediction for H5N1 Influenza

This repository contains implementations for predicting host adaptation in H5N1 PB2 influenza virus sequences using ESM-2 (Evolutionary Scale Modeling 2).

## Overview

This project analyzes H5N1 PB2 protein sequences to predict host types (human, avian, or other mammal) using two approaches:

1. **Baseline ESM-2** (`baseline_esm2.py`): Uses frozen pre-trained ESM-2 embeddings with classical machine learning classifiers
2. **Fine-tuned ESM-2** (`esm2_finetuning.py`): Fine-tunes ESM-2 with task-specific training for improved host prediction

Both approaches include comprehensive contact map analysis to identify structural differences associated with host adaptation.

## Repository Structure

```
project/
├── baseline_esm2.py              # Baseline model with frozen ESM-2 embeddings
├── esm2_finetuning.py           # Fine-tuned ESM-2 model implementation
├── baseline_results/            # Output directory for baseline results
├── finetune_results/            # Output directory for fine-tuning results
│   └── *_group_contact_diff.png # Contact map visualizations
├── data/                        # Dataset (fasta file)
└── README.md                    # This file
```

## Requirements

### Installation

```bash
# Create conda environment from .yml file
conda env create -f environment.yml
```

## Dataset Format

Input FASTA files should have headers in the following format:

```
>EPI4586304|PB2|A/dairy_cow/USA/019914-004/2025|EPI_ISL_20094668|A_/_H5N1|HOST_0
```

**Header Fields (pipe-separated):**
1. EPI ID
2. Gene name (e.g., PB2)
3. Strain information
4. EPI_ISL ID
5. Subtype (e.g., A_/_H5N1)
6. Host code:
   - `HOST_0` = Human
   - `HOST_1` = Avian
   - `HOST_2` = Other mammals

## Usage

### 1. Baseline ESM-2 Model

Runs pre-trained ESM-2 with frozen embeddings and classical classifiers (Logistic Regression, Random Forest).

```python
python baseline_esm2.py
```

**What it does:**
- Extracts sequence embeddings using pre-trained ESM-2
- Trains baseline classifiers on frozen features
- Computes group-level contact maps
- Generates visualizations and performance metrics

**Key Parameters (edit in `__main__` block):**
```python
fasta_path = 'path/to/your/sequences.fasta'
output_dir = 'baseline_results'
do_extract_attention_visuals = True  # Set False to skip attention analysis
n_attention_samples = 100  # Sequences per host type for attention analysis
```

**Outputs:**
- `metadata.csv` - Sequence metadata
- `esm2_sequence_embeddings.npz` - Compressed embeddings
- `confusion_matrix_*.png` - Classifier performance
- `contact_maps/` - Contact map visualizations
  - `human_avian_group_contact_diff.png`
  - `human_mammal_group_contact_diff.png`
  - `avian_mammal_group_contact_diff.png`
  - `contact_map_results.npz`
- `summary.txt` - Performance summary

### 2. Fine-tuned ESM-2 Model

Fine-tunes ESM-2 on the host prediction task with trainable layers.

```python
python esm2_finetuning.py
```

**What it does:**
- Fine-tunes ESM-2 with task-specific classification head
- Trains with early stopping based on validation performance
- Extracts contact maps from fine-tuned model
- Compares structural differences between host groups

**Key Configuration (edit `FineTuningConfig`):**
```python
config = FineTuningConfig(
    num_classes=3,
    model_name="esm2_t33_650M_UR50D",
    freeze_layers=28,      # Number of layers to freeze (0 = train all)
    hidden_dim=256,        # Classification head hidden dimension
    dropout=0.2,           # Dropout rate
    batch_size=8,          # Training batch size
    num_epochs=10,         # Maximum epochs
    learning_rate=1e-5,    # Learning rate
    val_size=0.2,          # Validation split
    test_size=0.2,         # Test split
    random_state=42        # Random seed
)
```

**Outputs:**
- `esm2_finetuned_checkpoint.pt` - Complete model checkpoint
- Contact map visualizations (3 pairwise comparisons)
- Network plots showing gained/lost contacts
- Comprehensive classification metrics

## Key Features

### Contact Map Analysis

Both scripts extract attention-based contact predictions from ESM-2 to analyze structural differences:

- **Group-level comparison**: Average contact maps for each host type
- **Effect size analysis**: Cohen's d for contact differences
- **Residue ranking**: Identifies positions with significant structural changes
- **Visualization**: Heatmaps, per-residue plots, and summary tables

### Classification Metrics

Comprehensive evaluation including:
- Accuracy (train/validation/test)
- Precision, Recall, F1-score (macro and weighted)
- Per-class metrics
- Confusion matrices


## Output Interpretation

### Contact Map Visualizations

The contact difference plots show:

1. **Top row (left to right)**:
   - Mean contact map for group 1
   - Mean contact map for group 2
   - Effect size heatmap (red = gained contacts, blue = lost contacts)

2. **Bottom row**:
   - Per-residue contact environment changes (bar plot)
   - Table of top 15 changed residues

**Interpretation:**
- High effect size positions indicate residues with structural environment changes
- These may be associated with host adaptation mechanisms
- Green highlights can mark known functional residues


## Citation

If you use this code, please cite:

**ESM-2:**
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yair and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

## License

This project is for academic and research purposes. Please comply with the ESM-2 model license and terms of use.

---

**Last Updated:** December 2024

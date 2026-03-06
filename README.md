# Morphological Modeling of Modern Standard Arabic (MSA) and Dialectal Arabic (DA) Using NLP with Deep Learning

**Author:** Mohamed Medouani  
**Year:** 2025

---

## Overview

This repository contains the complete implementation for the doctoral dissertation:

> *Morphological Modeling of Modern Standard Arabic (MSA) and Dialectal Arabic (DA) Using Natural Language Processing (NLP) with Deep Learning*

The study investigates how deep learning (DL) architectures can improve morphological modeling in Arabic NLP (ANLP), with a specific focus on addressing the performance gap between MSA-optimized tools and Dialectal Arabic (DA) varieties. Eight model architectures were implemented and evaluated across eight Arabic linguistic datasets covering MSA and five DA dialect clusters.

---

## Repository Structure

```
arabic_morph_analysis/
├── data/                          # Dataset loaders
│   ├── patb_loader.py             # MSA: Penn Arabic Treebank (PATB)
│   ├── gigaword_loader.py         # MSA: Arabic Gigaword 5th Edition
│   ├── sanad_loader.py            # MSA: SANAD Corpus
│   ├── madar_loader.py            # DA: MADAR Corpus (25 city-level dialects)
│   ├── egyptian_treebank_loader.py# DA: Egyptian Arabic Treebank (ARZ)
│   ├── lac_loader.py              # DA: Levantine Arabic Corpus (LAC)
│   ├── gac_loader.py              # DA: Gulf Arabic Corpus (GAC)
│   └── mad_loader.py              # DA: Maghrebi Arabic Dataset (MAD)
│
├── models/                        # Model implementations
│   ├── rule_based_model.py        # Baseline 1: Rule-Based (RB) — BAMA paradigm
│   ├── statistical_model.py       # Baseline 2: Statistical (SB) — MADA paradigm
│   ├── cl_bilstm_model.py         # Neural 1: Character-Level BiLSTM (CL-BiLSTM)
│   ├── wc_hybrid_model.py         # Neural 2: Word-Character Hybrid (WC-Hybrid)
│   ├── transformer_model.py       # Neural 3: Transformer-Based (AraBERT/MARBERT)
│   ├── dsm_model.py               # Multi-Dialect 1: Dialect-Specific Models (DSM)
│   ├── umd_model.py               # Multi-Dialect 2: Unified Multi-Dialect (UMD)
│   └── amd_model.py               # Multi-Dialect 3: Adaptive Multi-Dialect (AMD) ← BEST
│
├── utils/
│   ├── data_loader.py             # Base dataset class and DataLoader factory
│   └── metrics.py                 # All evaluation metrics (WLA, FLA, LA, SA, ARA, DTG, DRS, CSP)
│
├── run_benchmark.py               # Master benchmark script (train + evaluate + report)
├── config.yaml                    # Hyperparameter configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## The 8 Datasets

| # | Dataset | Type | Size | Dialect | Split |
|---|---------|------|------|---------|-------|
| 1 | **PATB** (Penn Arabic Treebank) | MSA | ~1.5M words | MSA | 80/10/10 |
| 2 | **Arabic Gigaword 5th Edition** | MSA | ~10M words (subset) | MSA | 90/10 |
| 3 | **SANAD Corpus** | MSA | ~1M words | MSA | Domain adaptation |
| 4 | **MADAR Corpus** | DA | ~12K sentences/dialect | 25 city-level dialects | 80/10/10 |
| 5 | **Egyptian Arabic Treebank (ARZ)** | DA | ~300K words | Egyptian | 80/10/10 |
| 6 | **Levantine Arabic Corpus (LAC)** | DA | ~200K words | Levantine | 80/10/10 |
| 7 | **Gulf Arabic Corpus (GAC)** | DA | ~150K words | Gulf | 75/10/15 |
| 8 | **Maghrebi Arabic Dataset (MAD)** | DA | ~100K words | Maghrebi | 80/20 |

> **Note:** Datasets are available from the Linguistic Data Consortium (LDC), the MADAR project, and the CAMeL Lab at NYU Abu Dhabi. Access requires institutional registration.

---

## The 8 Models

| # | Model | Type | Framework | Key Hyperparameters |
|---|-------|------|-----------|---------------------|
| 1 | **RB** (Rule-Based) | Baseline | Custom Python | BAMA-inspired lexicons |
| 2 | **SB** (Statistical) | Baseline | Custom Python | MaxEnt, char n-grams |
| 3 | **CL-BiLSTM** | Neural | PyTorch 1.9.0 | Embed: 128d, Hidden: 512, Layers: 2, Epochs: 30, Batch: 64 |
| 4 | **WC-Hybrid** | Neural | TensorFlow 2.6.0 | Word: 300d, Char: 128d, CNN + BiLSTM 512, Epochs: 25, Batch: 32 |
| 5 | **Trans** (AraBERT/MARBERT) | Transformer | HuggingFace 4.11.3 | Context: 512 tokens, Epochs: 10, Batch: 16, AdamW |
| 6 | **DSM** (Dialect-Specific) | Multi-Dialect | HuggingFace | Separate Trans model per dialect |
| 7 | **UMD** (Unified Multi-Dialect) | Multi-Dialect | HuggingFace | Single Trans + dialect embedding tokens |
| 8 | **AMD** (Adaptive Multi-Dialect) | Multi-Dialect | HuggingFace + PEFT | LoRA adapters: rank=8, alpha=32, dropout=0.1 |

---

## Key Results (Dissertation Chapter 4)

### Table 12: Word-Level Accuracy (WLA) on MSA Test Set

| Model | WLA (%) |
|-------|---------|
| RB (Rule-Based) | 82.1 |
| SB (Statistical) | 85.4 |
| CL-BiLSTM | 88.7 |
| WC-Hybrid | 90.2 |
| Trans (AraBERT) | **94.3** |

### Table 15: WLA Across Arabic Varieties (Best Model per Dialect)

| Dialect | Best Model | WLA (%) |
|---------|-----------|---------|
| MSA | Trans (AraBERT) | 94.3 |
| Egyptian | DSM (EGY) | 88.6 |
| Levantine | DSM (LEV) | 85.2 |
| Gulf | DSM (GULF) | 82.4 |
| Maghrebi | AMD | 74.1 |

### Table 20: Multi-Dialect Strategy Comparison

| Strategy | Avg. DA WLA (%) | DRS | Training Cost |
|----------|----------------|-----|---------------|
| DSM | 82.4 | 0.84 | High (×5 models) |
| UMD | 79.6 | 0.79 | Low |
| **AMD** | **84.1** | **0.91** | Medium |

> **Key Finding:** The Adaptive Multi-Dialect (AMD) model achieves the best balance of accuracy and robustness, with the highest Dialect Robustness Score (DRS: 0.91) and a 34% reduction in the Dialect Transfer Gap compared to standard fine-tuning.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **WLA** | Word-Level Accuracy — % of words where all features are correct |
| **FLA** | Feature-Level Accuracy — % of individual morphological features correct |
| **LA** | Lemmatization Accuracy |
| **SA** | Segmentation Accuracy |
| **ARA** | Ambiguity Resolution Accuracy |
| **DTG** | Dialect Transfer Gap — performance drop across dialects |
| **DRS** | Dialect Robustness Score — harmonic mean across all dialects |
| **CSP** | Code-Switching Performance |

---

## Environment

- **Python:** 3.8
- **GPU:** NVIDIA A100 (40GB)
- **Key Libraries:** PyTorch 1.9.0, TensorFlow 2.6.0, HuggingFace Transformers 4.11.3, CAMeL Tools 1.0.0, Farasa 0.3.0, PEFT (LoRA)

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download ar_core_news_sm
```

### 2. Prepare Data
Place your datasets in the `data/raw/` directory following the format described in each loader file.

### 3. Run the Benchmark
```bash
python run_benchmark.py --config config.yaml
```

To run a specific model, edit `config.yaml` and set `model_to_run` to one of: `RB`, `SB`, `CL_BiLSTM`, `WC_Hybrid`, `Trans`, `DSM`, `UMD`, `AMD`.

---

## Citation

If you use this code or the results from this study, please cite:

```bibtex
@phdthesis{medouani2025arabic,
  author  = {Mohamed Medouani},
  title   = {Morphological Modeling of Modern Standard Arabic (MSA) and Dialectal Arabic (DA) Using Natural Language Processing (NLP) with Deep Learning},
  school  = {Capitol Technology University},
  year    = {2025},
  type    = {Doctor of Science Dissertation}
}
```

---

## License

This repository is released for academic and research purposes. All datasets are subject to their respective license agreements (LDC, MADAR, CAMeL Lab).

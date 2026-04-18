**AI Hallucination Detection - Stacking Ensemble**
Overview
This repository contains a machine learning solution for detecting hallucinations in AI-generated scientific text. Given two AI-generated summaries of the same scientific paper, the model predicts which one contains factually incorrect information (hallucinations).

The solution uses a stacking ensemble of three specialized models trained on engineered linguistic features, perplexity scores, and TF-IDF representations. No large pretrained models (BERT/GPT) were used.

Key Results
Metric	Score
Cross-Validation Accuracy	92.47%
Kaggle Public LB	0.88737
Kaggle Private LB	0.88796
Model Architecture
text
                    ┌─────────────────┐
                    │   Input Pair    │
                    │ (Text A vs B)   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Model A     │    │   Model B     │    │   Model C     │
│ SVD + TF-IDF  │    │ Handcrafted   │    │  Linguistic   │
│ + Perplexity  │    │ + KL + PPL    │    │  + Perplexity │
│ (39 features) │    │ (84 features) │    │ (35 features) │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
              ┌─────────────────────────┐
              │   Meta-Model (Level 1)  │
              │   Logistic Regression   │
              └────────────┬────────────┘
                           ▼
                    ┌─────────────┐
                    │ Prediction  │
                    │ (Hallucinator│
                    │    or not)  │
                    └─────────────┘
Features
1. Linguistic Features (13 dimensions)
Burstiness (lexical distribution variance)

Hedging phrase density

Sentence length variance

Paragraph length variance

Hapax ratio (rare word proportion)

Punctuation diversity (quotes, exclamations, questions)

Bigram repetition ratio

2. Perplexity Features (3 dimensions)
Raw log-perplexity (KenLM 6-gram)

Sentence-averaged perplexity

OOV (Out-of-Vocabulary) rate

3. TF-IDF Features (compressed via SVD)
Word-level TF-IDF (500 features → 30 SVD components)

Character-level TF-IDF (500 features → 30 SVD components)

4. KL-Divergence Features (3 dimensions)
KL divergence from real/fake reference distributions

Reference Corpus
Source: arXiv abstracts (450,000 documents)

Filter: Pre-2024 only (to avoid AI contamination)

Model: 6-gram KenLM language model

Training Details
Validation: StratifiedGroupKFold (5-fold) with article-level grouping

Base Models: XGBRFClassifier, XGBClassifier, Logistic Regression

Ensemble: Stacking with Logistic Regression meta-model

Feature Engineering: Pairwise differences, absolute differences, and ratios

File Structure
text
├── notebooks/
│   └── improved_hallucination_detector_kenLM.ipynb   # Main notebook
├── data/
│   ├── train.csv                                       # Training labels
│   └── test/                                           # Test articles
├── models/
│   └── arxiv_6gram.arpa                                # KenLM model file
├── submissions/
│   └── submission_stacking.csv                        # Final predictions
└── README.md
Requirements
text
numpy
pandas
scikit-learn
xgboost
kenlm
matplotlib
seaborn
Usage
python
# Run the main notebook to:
# 1. Extract features from text pairs
# 2. Train base models and stacking ensemble
# 3. Generate predictions on test set

jupyter notebook improved_hallucination_detector_kenLM.ipynb
Key Insights
Feature Ablation	Accuracy Drop
All Linguistic Features	-8.2%
Perplexity Features	-6.7%
SVD Compression	-4.1%
Pairwise Ratios	-3.5%
KL-Divergence	-2.9%
Acknowledgments
KenLM library for efficient n-gram language modeling

arXiv for providing the reference corpus of scientific abstracts

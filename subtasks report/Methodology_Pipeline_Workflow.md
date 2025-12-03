# Methodology/Pipeline - Workflow and Block Diagrams

## 1. Overview

This document describes the complete methodology and pipeline for the SarcasmLens project, covering the end-to-end workflow from raw data collection to model training, evaluation, and deployment. The project employs multiple modeling approaches to comprehensively address sarcasm detection in code-mixed text.

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SarcasmLens System Architecture              │
└─────────────────────────────────────────────────────────────────┘

    Raw Datasets                    Data Preparation
    ┌─────────────┐                ┌──────────────┐
    │ Swami       │                │ Preprocessing│
    │ HackArena   │───────────────▶│ Normalization│
    │ Mendeley    │                │ Combination  │
    │ Aggarwal    │                └──────────────┘
    └─────────────┘                        │
                                            ▼
    Feature Extraction              ┌──────────────┐
    ┌─────────────┐                │ Combined     │
    │ TF-IDF      │                │ Dataset      │
    │ FastText    │                │ (15,099)     │
    │ Tokenization│                └──────────────┘
    └─────────────┘                        │
                                            ▼
    Model Training                  ┌──────────────┐
    ┌─────────────┐                │ Train/Test   │
    │ Traditional │                │ Split        │
    │ ML Models   │                │ (80/20)      │
    │ Deep Learning│               └──────────────┘
    │ Transformers│                        │
    └─────────────┘                        ▼
                                    ┌──────────────┐
    Evaluation                      │ Model        │
    ┌─────────────┐                │ Training     │
    │ Metrics     │                │ & Evaluation │
    │ Comparison  │                └──────────────┘
    └─────────────┘                        │
                                            ▼
                                    ┌──────────────┐
                                    │ Best Model   │
                                    │ Selection    │
                                    └──────────────┘
```

## 3. Complete Workflow

### 3.1 Phase 1: Data Collection and Preparation

**Objective**: Collect and prepare code-mixed sarcasm datasets from multiple sources.

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Data Collection and Preparation                    │
└─────────────────────────────────────────────────────────────┘

Step 1.1: Raw Data Collection
├── Swami Dataset (TXT files)
│   ├── Sarcasm_tweets.txt
│   └── Sarcasm_tweet_truth.txt
│
├── HackArena Dataset (CSV files)
│   ├── train.csv
│   └── test.csv
│
├── Mendeley Dataset (CSV with sentiment)
│   └── labeled_sentiment_output_codemixed.csv
│
└── Aggarwal Dataset (CSV with tweet IDs)
    └── Sarcasm_dataset.csv

Step 1.2: Data Conversion
├── create_csv.py
│   └── Converts Swami TXT → CSV format
│
├── convert_labels.py
│   └── Normalizes labels (YES/NO → 1/0)
│
├── extract_text_and_sentiment.py
│   └── Extracts text and converts sentiment to labels
│
└── fetch_tweets.py (optional)
    └── Fetches full tweet text from API

Step 1.3: Data Preprocessing
└── preprocess_data.py
    ├── Lowercase conversion
    ├── Remove @ mentions
    ├── Process hashtags (camelCase splitting)
    ├── Remove URLs
    ├── Remove punctuation
    ├── Remove English stop words
    └── Normalize whitespace

Step 1.4: Dataset Combination
└── combine_datasets.py
    ├── Extract text and label columns
    ├── Remove missing/empty values
    ├── Merge all datasets
    └── Output: combined_dataset.csv (15,099 samples)
```

### 3.2 Phase 2: Feature Extraction and Representation

**Objective**: Transform preprocessed text into numerical representations suitable for machine learning models.

#### 2.1 TF-IDF Vectorization Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ TF-IDF Feature Extraction Pipeline                           │
└─────────────────────────────────────────────────────────────┘

Preprocessed Text
        │
        ▼
┌───────────────┐
│ TF-IDF        │
│ Vectorizer    │
│               │
│ Parameters:   │
│ - max_features│
│ - ngram_range │
│ - min_df      │
│ - max_df      │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Sparse Matrix │
│ (n_samples ×  │
│  n_features)  │
└───────────────┘
        │
        ▼
    Model Input
```

**Parameters**:
- `max_features`: 5000 (vocabulary size)
- `ngram_range`: (1, 2) - unigrams and bigrams
- `min_df`: 2 (minimum document frequency)
- `max_df`: 0.95 (maximum document frequency)
- `stop_words`: None (preserved for sarcasm detection)

#### 2.2 FastText Embedding Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ FastText Embedding Pipeline                                  │
└─────────────────────────────────────────────────────────────┘

All Text Data
        │
        ▼
┌───────────────┐
│ Prepare       │
│ FastText      │
│ Training Data │
│ (one sentence │
│  per line)    │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Train         │
│ FastText      │
│ Model         │
│               │
│ Parameters:   │
│ - dim: 100    │
│ - model:      │
│   skipgram    │
│ - epoch: 10   │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Extract Word  │
│ Embeddings    │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Sentence      │
│ Embedding     │
│ (Average word │
│  vectors)     │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Dense Matrix  │
│ (n_samples ×  │
│  100)         │
└───────────────┘
        │
        ▼
    Model Input
```

**Process**:
1. Train unsupervised FastText model on all text data
2. Extract word embeddings for each word in vocabulary
3. Compute sentence embeddings by averaging word vectors
4. Output: Dense feature matrix (n_samples × 100)

#### 2.3 Sequence Tokenization Pipeline (BiLSTM)

```
┌─────────────────────────────────────────────────────────────┐
│ Sequence Tokenization Pipeline (BiLSTM)                     │
└─────────────────────────────────────────────────────────────┘

Preprocessed Text
        │
        ▼
┌───────────────┐
│ Keras         │
│ Tokenizer     │
│               │
│ Parameters:   │
│ - vocab_size: │
│   20000       │
│ - oov_token:│
│   <OOV>       │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Text to       │
│ Sequences     │
│ (Integer IDs) │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Padding &     │
│ Truncation    │
│               │
│ - max_len:    │
│   95th        │
│   percentile  │
│ - padding:    │
│   post        │
│ - truncating: │
│   post        │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Padded        │
│ Sequences     │
│ (n_samples ×  │
│  max_len)     │
└───────────────┘
        │
        ▼
    Model Input
```

#### 2.4 Transformer Tokenization Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Transformer Tokenization Pipeline                           │
└─────────────────────────────────────────────────────────────┘

Preprocessed Text
        │
        ▼
┌───────────────┐
│ AutoTokenizer │
│ (Model-       │
│  specific)    │
│               │
│ Models:       │
│ - XLM-RoBERTa │
│ - mBERT       │
│ - Indic-BERT  │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Tokenize      │
│               │
│ Parameters:   │
│ - truncation: │
│   True        │
│ - padding:    │
│   max_length  │
│ - max_length: │
│   96          │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Tokenized     │
│ Dataset       │
│               │
│ Fields:       │
│ - input_ids   │
│ - attention_  │
│   mask        │
│ - labels      │
└───────────────┘
        │
        ▼
    Model Input
```

### 3.3 Phase 3: Model Training Pipelines

#### 3.1 Traditional Machine Learning Pipeline (TF-IDF + Classifiers)

```
┌─────────────────────────────────────────────────────────────┐
│ Traditional ML Pipeline (TF-IDF + Classifiers)             │
└─────────────────────────────────────────────────────────────┘

Combined Dataset
        │
        ▼
┌───────────────┐
│ Train/Test    │
│ Split         │
│ (80/20,       │
│  stratified) │
└───────────────┘
        │
        ├─────────────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────┐
│ Train Set     │  │ Test Set      │
│ (80%)         │  │ (20%)         │
└───────────────┘  └───────────────┘
        │                 │
        ▼                 │
┌───────────────┐        │
│ TF-IDF        │        │
│ Vectorization │        │
│ (fit on train)│        │
└───────────────┘        │
        │                 │
        ├─────────────────┤
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────┐
│ Train TF-IDF  │  │ Test TF-IDF   │
│ Features      │  │ Features      │
│ (sparse)      │  │ (sparse)      │
└───────────────┘  └───────────────┘
        │                 │
        ▼                 │
┌───────────────┐        │
│ Train Model   │        │
│               │        │
│ Options:      │        │
│ - Random      │        │
│   Forest      │        │
│ - SVM         │        │
└───────────────┘        │
        │                 │
        └─────────────────┘
                 │
                 ▼
        ┌───────────────┐
        │ Predictions   │
        │ on Test Set   │
        └───────────────┘
                 │
                 ▼
        ┌───────────────┐
        │ Evaluation    │
        │ Metrics       │
        │ - Accuracy   │
        │ - F1 Score    │
        │ - Precision   │
        │ - Recall      │
        └───────────────┘
```

**Models**:
- **Random Forest**: `n_estimators=100`, `class_weight='balanced'`
- **SVM**: `C=1.0`, `max_iter=1000`, `class_weight='balanced'`

#### 3.2 FastText Embedding + Classifier Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ FastText Embedding + Classifier Pipeline                     │
└─────────────────────────────────────────────────────────────┘

Combined Dataset
        │
        ▼
┌───────────────┐
│ Train/Test    │
│ Split         │
│ (80/20)       │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Train FastText│
│ Model on ALL  │
│ Data          │
│ (unsupervised)│
└───────────────┘
        │
        ├─────────────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────┐
│ Extract Train │  │ Extract Test  │
│ Embeddings    │  │ Embeddings    │
│ (avg word     │  │ (avg word     │
│  vectors)     │  │  vectors)     │
└───────────────┘  └───────────────┘
        │                 │
        ▼                 │
┌───────────────┐        │
│ Train         │        │
│ Classifier    │        │
│               │        │
│ Options:      │        │
│ - Random      │        │
│   Forest      │        │
│ - SVM         │        │
└───────────────┘        │
        │                 │
        └─────────────────┘
                 │
                 ▼
        ┌───────────────┐
        │ Predictions   │
        │ & Evaluation │
        └───────────────┘
```

#### 3.3 Deep Learning Pipeline (BiLSTM)

```
┌─────────────────────────────────────────────────────────────┐
│ Deep Learning Pipeline (BiLSTM)                              │
└─────────────────────────────────────────────────────────────┘

Combined Dataset
        │
        ▼
┌───────────────┐
│ Train/Test    │
│ Split         │
│ (80/20)       │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Tokenization  │
│ (Keras        │
│  Tokenizer)   │
│ - Fit on train│
│ - vocab_size: │
│   20000       │
└───────────────┘
        │
        ├─────────────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────┐
│ Train         │  │ Test          │
│ Sequences     │  │ Sequences     │
│ (padded)      │  │ (padded)      │
└───────────────┘  └───────────────┘
        │                 │
        ▼                 │
┌───────────────┐        │
│ Build BiLSTM  │        │
│ Model         │        │
│               │        │
│ Architecture: │        │
│ - Embedding   │        │
│ - BiLSTM      │        │
│ - Pooling     │        │
│ - Dense       │        │
│ - Output      │        │
└───────────────┘        │
        │                 │
        ▼                 │
┌───────────────┐        │
│ Training      │        │
│               │        │
│ - Optimizer:  │        │
│   Adam        │        │
│ - Loss:       │        │
│   binary_     │        │
│   crossentropy│        │
│ - Callbacks:  │        │
│   EarlyStop,  │        │
│   Checkpoint  │        │
└───────────────┘        │
        │                 │
        └─────────────────┘
                 │
                 ▼
        ┌───────────────┐
        │ Predictions   │
        │ & Evaluation │
        └───────────────┘
```

**Model Architecture**:
```
Input (padded sequences)
    │
    ▼
Embedding Layer (vocab_size × 128)
    │
    ▼
Bidirectional LSTM (64 units, return_sequences=True)
    │
    ▼
Global Max Pooling
    │
    ▼
Dropout (0.5)
    │
    ▼
Dense (64, ReLU)
    │
    ▼
Dropout (0.5)
    │
    ▼
Dense (1, Sigmoid) → Output (0 or 1)
```

#### 3.4 Transformer Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Transformer Pipeline                                         │
└─────────────────────────────────────────────────────────────┘

Combined Dataset
        │
        ▼
┌───────────────┐
│ Train/Val/Test│
│ Split         │
│ (80/10/10)    │
└───────────────┘
        │
        ├──────────┬──────────┐
        │          │          │
        ▼          ▼          ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│ Train     │ │ Validation│ │ Test       │
│ Set       │ │ Set       │ │ Set       │
└───────────┘ └───────────┘ └───────────┘
        │          │          │
        └──────────┴──────────┘
                 │
                 ▼
┌───────────────────────────────┐
│ Tokenization                  │
│ (Model-specific tokenizer)     │
│                               │
│ Models:                       │
│ - xlm-roberta-base            │
│ - bert-base-multilingual-cased│
│ - ai4bharat/indic-bert        │
└───────────────────────────────┘
        │
        ├──────────┬──────────┐
        │          │          │
        ▼          ▼          ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│ Train     │ │ Val       │ │ Test      │
│ Tokens    │ │ Tokens    │ │ Tokens    │
└───────────┘ └───────────┘ └───────────┘
        │          │          │
        ▼          │          │
┌───────────────┐ │          │
│ Load Pre-     │ │          │
│ trained Model │ │          │
│ (AutoModelFor │ │          │
│  Sequence     │ │          │
│  Classification)│          │
└───────────────┘ │          │
        │          │          │
        ▼          │          │
┌───────────────┐ │          │
│ Fine-tuning   │ │          │
│               │ │          │
│ Training Args:│ │          │
│ - lr: 2e-5    │ │          │
│ - batch: 8    │ │          │
│ - epochs: 3   │ │          │
│ - eval on val │ │          │
└───────────────┘ │          │
        │          │          │
        └──────────┴──────────┘
                 │
                 ▼
        ┌───────────────┐
        │ Evaluate on   │
        │ Test Set      │
        └───────────────┘
                 │
                 ▼
        ┌───────────────┐
        │ Compare All   │
        │ Models         │
        └───────────────┘
```

**Training Configuration**:
- **Learning Rate**: 2e-5
- **Batch Size**: 8 (train), 16 (eval)
- **Epochs**: 3
- **Max Length**: 96 tokens
- **Optimization**: AdamW with weight decay
- **Evaluation**: Per epoch on validation set
- **Metric**: F1 score (best model selection)

### 3.4 Phase 4: Evaluation and Comparison

```
┌─────────────────────────────────────────────────────────────┐
│ Evaluation Pipeline                                          │
└─────────────────────────────────────────────────────────────┘

All Trained Models
        │
        ▼
┌───────────────┐
│ Test Set      │
│ Predictions   │
│ (for each     │
│  model)       │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Compute       │
│ Metrics       │
│               │
│ - Accuracy    │
│ - F1 Score    │
│ - Precision   │
│ - Recall      │
│ - Confusion    │
│   Matrix      │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Model         │
│ Comparison    │
│ Table         │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Best Model    │
│ Selection     │
│ (FastText +   │
│  Random Forest)│
└───────────────┘
```

## 4. Complete System Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Complete SarcasmLens Workflow                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ Raw Datasets │
│ (4 sources)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Data Preparation Phase              │
│                                     │
│  ┌──────────────┐                  │
│  │ Conversion   │                  │
│  │ Scripts      │                  │
│  └──────┬───────┘                  │
│         │                           │
│         ▼                           │
│  ┌──────────────┐                  │
│  │ Preprocessing │                  │
│  │ Pipeline      │                  │
│  └──────┬───────┘                  │
│         │                           │
│         ▼                           │
│  ┌──────────────┐                  │
│  │ Combination  │                  │
│  │ Script       │                  │
│  └──────┬───────┘                  │
└─────────┼───────────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Combined Dataset    │
│ (15,099 samples)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Train/Test Split (80/20)            │
└──────────┬──────────────────────────┘
           │
           ├──────────────────────────────┐
           │                              │
           ▼                              ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Feature Extraction   │    │ Feature Extraction   │
│                      │    │                      │
│ ┌──────────────────┐ │    │ ┌──────────────────┐ │
│ │ TF-IDF           │ │    │ │ FastText         │ │
│ │ Vectorization    │ │    │ │ Embeddings       │ │
│ └────────┬─────────┘ │    │ └────────┬─────────┘ │
│          │           │    │          │           │
│          ▼           │    │          ▼           │
│ ┌──────────────────┐ │    │ ┌──────────────────┐ │
│ │ Sparse Matrix   │ │    │ │ Dense Matrix    │ │
│ │ (5000 features)  │ │    │ │ (100 features)  │ │
│ └────────┬─────────┘ │    │ └────────┬─────────┘ │
└──────────┼────────────┘    └──────────┼────────────┘
           │                            │
           │                            │
           ├────────────────────────────┼──────────────┐
           │                            │              │
           ▼                            ▼              ▼
┌──────────────────┐    ┌──────────────────┐  ┌──────────────────┐
│ Model Training   │    │ Model Training   │  │ Model Training   │
│                  │    │                  │  │                  │
│ TF-IDF + RF      │    │ FastText + RF    │  │ BiLSTM           │
│ TF-IDF + SVM     │    │ FastText + SVM   │  │                  │
└────────┬─────────┘    └────────┬─────────┘  └────────┬─────────┘
         │                        │                     │
         └────────────────────────┼─────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │ Model Evaluation     │
                    │                      │
                    │ - Accuracy           │
                    │ - F1 Score           │
                    │ - Precision/Recall    │
                    │ - Confusion Matrix    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Comparison     │
                    │                      │
                    │ Best: FastText + RF  │
                    │ (97.72% accuracy)   │
                    └──────────────────────┘
```

## 5. Model-Specific Workflows

### 5.1 TF-IDF + Random Forest Workflow

```
Input Text → Preprocessing → TF-IDF Vectorization → Random Forest → Prediction
                                                          │
                                                          ▼
                                                    Feature Importance
                                                    Analysis
```

**Steps**:
1. Preprocess text (lowercase, remove mentions, etc.)
2. Fit TF-IDF vectorizer on training data
3. Transform train and test sets
4. Train Random Forest classifier
5. Make predictions
6. Evaluate and analyze feature importance

### 5.2 FastText + Random Forest Workflow

```
All Text Data → FastText Training → Word Embeddings → Sentence Embeddings → Random Forest → Prediction
```

**Steps**:
1. Prepare FastText training data (all text)
2. Train unsupervised FastText model
3. Extract word embeddings for each word
4. Compute sentence embeddings (average word vectors)
5. Train Random Forest on sentence embeddings
6. Evaluate on test set

### 5.3 BiLSTM Workflow

```
Text → Tokenization → Sequence Padding → Embedding Layer → BiLSTM → Pooling → Dense Layers → Prediction
                                                                                              │
                                                                                              ▼
                                                                                        Training History
                                                                                        & Metrics
```

**Steps**:
1. Tokenize text to integer sequences
2. Pad sequences to fixed length
3. Build BiLSTM model architecture
4. Train with early stopping and checkpointing
5. Evaluate on test set
6. Analyze training history

### 5.4 Transformer Workflow

```
Text → Tokenization (Model-specific) → Pre-trained Model → Fine-tuning → Evaluation → Comparison
                                                                              │
                                                                              ▼
                                                                        Best Model
                                                                        Selection
```

**Steps**:
1. Load pre-trained transformer model and tokenizer
2. Tokenize text with model-specific tokenizer
3. Fine-tune on training set with validation monitoring
4. Evaluate on test set
5. Compare all three transformer models
6. Select best performing model

## 6. Evaluation Methodology

### 6.1 Metrics

All models are evaluated using the following metrics:

1. **Accuracy**: Overall correctness
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **F1 Score**: Harmonic mean of precision and recall
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

3. **Precision**: Proportion of positive predictions that are correct
   ```
   Precision = TP / (TP + FP)
   ```

4. **Recall**: Proportion of actual positives correctly identified
   ```
   Recall = TP / (TP + FN)
   ```

5. **Confusion Matrix**: Detailed breakdown of predictions
   ```
   ┌─────────────┬─────────────┐
   │             │  Predicted  │
   ├─────────────┼─────────────┤
   │             │  0     1    │
   ├─────────────┼─────────────┤
   │ Actual   0  │ TN     FP   │
   │         1  │ FN     TP   │
   └─────────────┴─────────────┘
   ```

### 6.2 Evaluation Process

```
For each model:
    1. Load trained model
    2. Make predictions on test set
    3. Compute all metrics
    4. Generate confusion matrix
    5. Print classification report
    6. Save results

Compare all models:
    1. Create comparison table
    2. Identify best model
    3. Analyze performance differences
    4. Document findings
```

### 6.3 Model Comparison Results

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| TF-IDF + Random Forest | 0.9755 | 0.9755 | - | - |
| TF-IDF + SVM | 0.9762 | 0.9762 | - | - |
| **FastText + Random Forest** | **0.9772** | **0.9771** | - | - |
| FastText + SVM | 0.9493 | 0.9492 | - | - |
| BiLSTM | 0.9705 | 0.9705 | - | - |
| XLM-RoBERTa | - | - | - | - |
| mBERT | - | - | - | - |
| Indic-BERT | - | - | - | - |

*Note: Transformer results are pending training completion.*

## 7. Implementation Details

### 7.1 Technology Stack

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation
- **Scikit-learn**: Traditional ML models and metrics
- **NLTK**: Text preprocessing
- **FastText**: Word embeddings
- **TensorFlow/Keras**: Deep learning (BiLSTM)
- **PyTorch**: Transformer models
- **Transformers (HuggingFace)**: Pre-trained models

### 7.2 Reproducibility

- **Random Seeds**: Fixed seed (42) for all random operations
- **Stratified Splits**: Maintains class distribution
- **Version Control**: All scripts and configurations tracked
- **Documentation**: Complete pipeline documented

### 7.3 Model Persistence

All trained models and artifacts are saved:
- **TF-IDF Models**: `svm_model.pkl`, `random_forest_model.pkl`
- **Vectorizers**: `tfidf_vectorizer.pkl`, `tfidf_vectorizer_svm.pkl`
- **FastText Model**: `fasttext_model.bin`
- **FastText Classifiers**: `random_forest_fasttext_model.pkl`, `svm_fasttext_model.pkl`
- **BiLSTM Model**: `bilstm_model.keras`
- **BiLSTM Tokenizer**: `tokenizer_bilstm.pkl`
- **Transformer Models**: Saved in `transformer_runs/` directory

## 8. Pipeline Execution Order

### Recommended Execution Sequence

```
1. Data Preparation
   ├── Run create_csv.py (if needed)
   ├── Run convert_labels.py (if needed)
   ├── Run extract_text_and_sentiment.py (if needed)
   ├── Run preprocess_data.py on each dataset
   └── Run combine_datasets.py

2. Model Training (can run in parallel)
   ├── train_random_forest.py
   ├── train_svm.py
   ├── train_fasttext_models.py
   ├── train_bilstm.py
   └── train_transformer.py

3. Evaluation
   └── Compare results from all models
```

## 9. Conclusion

The SarcasmLens methodology employs a comprehensive, multi-approach pipeline that:

1. **Handles Multiple Data Sources**: Robust data collection and combination
2. **Applies Diverse Feature Extraction**: TF-IDF, FastText, tokenization
3. **Trains Various Models**: Traditional ML, embeddings, deep learning, transformers
4. **Ensures Rigorous Evaluation**: Multiple metrics and comparison framework
5. **Maintains Reproducibility**: Fixed seeds, stratified splits, documentation

This multi-faceted approach allows for comprehensive comparison and identification of the most effective method for sarcasm detection in code-mixed text, with the FastText + Random Forest model currently achieving the best performance at 97.72% accuracy.

---

*This document describes the complete methodology and pipeline workflow for the SarcasmLens project, covering all phases from data collection to model evaluation.*


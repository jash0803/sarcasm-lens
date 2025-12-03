# SarcasmLens – Subjective Sarcasm Detection in Code-Mixed Text

A comprehensive machine learning system for detecting sarcasm in Hindi-English code-mixed social media text using multiple approaches including traditional ML models, deep learning, and transformer-based architectures.

## Problem Statement

Sarcasm and irony in Indian social-media posts often emerge through subtle cues such as exaggeration, code-mixing of Hindi and English, emotive punctuation, or cultural references. Detecting such subjectivity poses unique linguistic and contextual challenges. 

The task is to create a system that can automatically identify sarcastic expressions in code-mixed social-media text while capturing the underlying subjectivity that distinguishes sarcasm from ordinary sentiment.

## Objectives

1. Investigate linguistic patterns of sarcasm in multilingual and code-mixed environments.
2. Construct an appropriate representation of text that preserves both language and stylistic information.
3. Design and justify an approach for distinguishing literal from sarcastic statements under noisy, informal text conditions.
4. Reflect on the rationale for data selection, model design, and evaluation strategy.

## Model Performance

The following table compares the performance of different models trained on the combined dataset:

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| TF-IDF + Random Forest | 0.9755 | 0.9755 | 0.9754 | 0.9756 |
| TF-IDF + SVM | 0.9762 | 0.9762 | 0.9761 | 0.9763 |
| FastText + Random Forest | **0.9772** | **0.9771** | **0.9779** | 0.9772 |
| FastText + SVM | 0.9493 | 0.9492 | 0.9509 | 0.9493 |
| BiLSTM | 0.9705 | 0.9705 | 0.9704 | 0.9706 |
| XLM-RoBERTa-base | 0.9669 | 0.9652 | 0.9429 | 0.9886 |
| mBERT | 0.9609 | 0.9587 | 0.9421 | 0.9757 |
| Indic-BERT | 0.9649 | 0.9633 | 0.9355 | **0.9929** |

The **FastText + Random Forest** model achieved the best overall performance with an accuracy of 97.72%, F1 score of 97.71%, and precision of 97.79%. Among transformer models, **XLM-RoBERTa-base** achieved the highest accuracy (96.69%) and F1 score (96.52%), while **Indic-BERT** achieved the highest recall (99.29%).

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd sarcasm-lens
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (required for preprocessing):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Project Structure

```
sarcasm-lens/
├── datasets/                    # Dataset files
│   ├── combined_dataset.csv     # Combined preprocessed dataset
│   ├── hackarena/              # HackArena dataset
│   ├── mendeley data/          # Mendeley dataset
│   ├── sahil swami/            # Sahil Swami dataset
│   └── akshita agrawall/       # Akshita Aggarwal dataset
├── scripts/
│   ├── data-preprocessing/     # Data preprocessing scripts
│   │   ├── combine_datasets.py
│   │   ├── convert_labels.py
│   │   ├── create_csv.py
│   │   ├── fetch_tweets.py
│   │   └── preprocess_data.py
│   └── models/                 # Model training scripts
│       ├── train_bilstm.py
│       ├── train_fasttext_models.py
│       ├── train_random_forest.py
│       ├── train_svm.py
│       └── train_transformer.py
├── saved-models/               # Trained model files
├── transformer_runs/           # Transformer model checkpoints
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

### Data Preprocessing

1. **Preprocess individual datasets:**
```bash
python scripts/dataset/preprocess_data.py \
    --input_file "datasets/sahil swami/sarcasm_dataset.csv" \
    --output_file "datasets/sahil swami/sahil_swami_processed.csv" \
    --text_column "text"
```

2. **Combine multiple datasets:**
```bash
python scripts/dataset/combine_datasets.py \
    --input_files "datasets/sahil swami/sahil_swami_processed.csv,datasets/hackarena/hackarena_train_processed.csv" \
    --output_file "datasets/combined_dataset.csv" \
    --text_column "text" \
    --label_column "label"
```

### Model Training

#### Traditional ML Models (TF-IDF based)

1. **Train Random Forest:**
```bash
python scripts/models/train_random_forest.py
```

2. **Train SVM:**
```bash
python scripts/models/train_svm.py
```

#### FastText-based Models

Train both Random Forest and SVM with FastText embeddings:
```bash
python scripts/models/train_fasttext_models.py
```

#### Deep Learning Models

1. **Train BiLSTM:**
```bash
python scripts/models/train_bilstm.py
```

#### Transformer Models

Train transformer models (XLM-RoBERTa, mBERT, or Indic-BERT):
```bash
# Train XLM-RoBERTa-base
python scripts/models/train_transformer.py -m xlm-roberta-base

# Train mBERT
python scripts/models/train_transformer.py -m mbert

# Train Indic-BERT
python scripts/models/train_transformer.py -m indic-bert

# Or use any Hugging Face model ID directly
python scripts/models/train_transformer.py -m bert-base-multilingual-cased
```

### Model Outputs

- **Traditional ML models**: Saved as `.pkl` files in `saved-models/`
- **BiLSTM**: Saved as `bilstm_model.keras` with tokenizer and config files
- **FastText**: Saved as `fasttext_model.bin`
- **Transformers**: Saved as checkpoints in `transformer_runs/{model_name}/checkpoint-{step}/`

## Dataset Information

The project uses multiple datasets for training:

1. **Sahil Swami Dataset**: English-Hindi code-mixed tweets for sarcasm detection
2. **HackArena Dataset**: Multilingual sarcasm detection dataset from Kaggle
3. **Mendeley Dataset**: Code-mixed social media content with sentiment labels
4. **Akshita Aggarwal Dataset**: Hindi-English code-mixed tweets

All datasets are combined and preprocessed to create a unified training dataset (`datasets/combined_dataset.csv`).

## Model Details

### Traditional ML Models
- **TF-IDF Vectorization**: Unigrams and bigrams with max features of 5000
- **Random Forest**: 100 estimators with balanced class weights
- **SVM**: Linear SVM with C=1.0 and balanced class weights

### FastText Models
- **Embedding Dimension**: 100
- **Training**: Supervised FastText model trained on the combined dataset
- **Classifiers**: Random Forest and SVM on FastText embeddings

### Deep Learning Models
- **BiLSTM**: Bidirectional LSTM with embedding layer, dropout, and early stopping
- **Architecture**: Embedding → BiLSTM → Dropout → Dense → Output

### Transformer Models
- **XLM-RoBERTa-base**: Multilingual transformer model
- **mBERT**: Multilingual BERT model
- **Indic-BERT**: BERT model specifically trained for Indian languages
- **Training**: Fine-tuned with learning rate 2e-5, batch size 8, 1 epoch

## References

1. Swami, S., Khandelwal, A., Singh, V., Akhtar, S. S., & Shrivastava, M.  
   *A Corpus of English-Hindi Code-Mixed Tweets for Sarcasm Detection*.  
   Available at: [https://arxiv.org/abs/1805.11869](https://arxiv.org/abs/1805.11869)

2. Aggarwal, A., Wadhawan, A., Chaudhary, A., & Maurya, K.  
   *"Did you really mean what you said?" : Sarcasm Detection in Hindi-English Code-Mixed Data using Bilingual Word Embeddings*.  
   Available at: [https://arxiv.org/abs/2010.00310](https://arxiv.org/abs/2010.00310)

3. Kaggle Dataset:  
   *HackArena Theme 2 – Multilingual Sarcasm Detection*.  
   Available at: [https://www.kaggle.com/datasets/divyanshu134/hackarena-theme-2-multilingual-sarcasm-detection/data](https://www.kaggle.com/datasets/divyanshu134/hackarena-theme-2-multilingual-sarcasm-detection/data)

4. Bedi, M., Kumar, S., Akhtar, M. S., & Chakraborty, T.  
   *Multi-modal Sarcasm Detection and Humor Classification in Code-mixed Conversations*.  
   Available at: [https://arxiv.org/abs/2105.09984](https://arxiv.org/abs/2105.09984)
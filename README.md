# SarcasmLens – Subjective Sarcasm Detection in Code-Mixed Text

## Problem Statement: 
Sarcasm and irony in Indian social-media posts often emerge through subtle cues such as exaggeration, code-mixing of Hindi and English, emotive punctuation, or cultural references. Detecting such subjectivity poses unique linguistic and contextual challenges. 
The task is to create a system that can automatically identify sarcastic expressions in code-mixed social-media text while capturing the underlying subjectivity that distinguishes sarcasm from ordinary sentiment.

## Objectives : 
1. Investigate linguistic patterns of sarcasm in multilingual and code-mixed
environments.
2. Construct an appropriate representation of text that preserves both language and
stylistic information.
3. Design and justify an approach for distinguishing literal from sarcastic statements
under noisy, informal text conditions.
4. Reflect on the rationale for data selection, model design, and evaluation strategy.

## Model Performance

The following table compares the performance of different models trained on the combined dataset:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + Random Forest | 0.9755 | 0.9755 |
| TF-IDF + SVM | 0.9762 | 0.9762 |
| FastText + Random Forest | **0.9772** | **0.9771** |
| FastText + SVM | 0.9493 | 0.9492 |
| BiLSTM (FastText) | 0.9705 | 0.9705 |

The FastText + Random Forest model achieved the best performance with an accuracy of 97.72% and F1 score of 97.71%.

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
# Model Framework - Architecture, Algorithms, and Design Choices

## 1. Overview

This document describes the model architectures, underlying algorithms, and design choices for the SarcasmLens project. The project employs a multi-faceted approach, comparing traditional machine learning, embedding-based, deep learning, and transformer-based models to identify the most effective method for sarcasm detection in code-mixed text.

## 2. Model Categories

The project implements models across four categories:

1. **Traditional ML with TF-IDF**: TF-IDF vectorization + Random Forest/SVM
2. **Embedding-Based**: FastText embeddings + Random Forest/SVM
3. **Deep Learning**: Bidirectional LSTM (BiLSTM)
4. **Transformer-Based**: Fine-tuned pre-trained transformers (XLM-RoBERTa, mBERT, Indic-BERT)

## 3. Feature Extraction Methods

### 3.1 TF-IDF (Term Frequency-Inverse Document Frequency)

#### Algorithm

TF-IDF is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents.

**Term Frequency (TF)**:
```
TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```

**Inverse Document Frequency (IDF)**:
```
IDF(t, D) = log(Total number of documents / Number of documents containing term t)
```

**TF-IDF Score**:
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

#### Architecture

```
Input Text
    │
    ▼
┌──────────────────┐
│ Text             │
│ Preprocessing    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tokenization     │
│ (Unigrams +      │
│  Bigrams)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ TF-IDF           │
│ Computation      │
│                  │
│ - max_features:  │
│   5000           │
│ - ngram_range:   │
│   (1, 2)         │
│ - min_df: 2      │
│ - max_df: 0.95   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Sparse Matrix    │
│ (n_samples ×     │
│  5000 features)  │
└──────────────────┘
```

#### Design Choices

**max_features = 5000**:
- **Rationale**: Limits vocabulary size to most frequent terms, reducing dimensionality and computational cost
- **Trade-off**: May lose rare but informative terms; balances performance and efficiency

**ngram_range = (1, 2)**:
- **Rationale**: Captures both individual words (unigrams) and word pairs (bigrams)
- **Benefit**: Bigrams capture contextual patterns important for sarcasm (e.g., "sure thing", "obviously great")
- **Trade-off**: Increases feature space but captures more context

**min_df = 2**:
- **Rationale**: Filters out terms appearing in only one document (likely typos or noise)
- **Benefit**: Reduces noise and improves generalization

**max_df = 0.95**:
- **Rationale**: Removes terms appearing in >95% of documents (likely not discriminative)
- **Benefit**: Focuses on terms that distinguish between classes

**stop_words = None**:
- **Rationale**: Stop words can be important for sarcasm (e.g., "really", "very", "sure")
- **Design Choice**: Preserves all words to capture sarcastic markers

### 3.2 FastText Embeddings

#### Algorithm

FastText uses a skip-gram model with character-level n-grams to learn word representations.

**Skip-gram Objective**:
```
Maximize: Σ log P(w_{t+j} | w_t)
```

Where the probability is computed using:
```
P(w_O | w_I) = exp(v'_{w_O}^T v_{w_I}) / Σ_{w=1}^W exp(v'_w^T v_{w_I})
```

**Character n-grams**:
- Each word is represented as a bag of character n-grams
- Example: "hello" → {<h, he, el, ll, lo, o>, <he, ell, llo>, <hel, ello, llo>}
- Handles out-of-vocabulary words and morphological variations

**Sentence Embedding**:
```
sentence_vector = (1/n) × Σ word_vector_i
```
Average of all word vectors in the sentence.

#### Architecture

```
All Text Data
    │
    ▼
┌──────────────────┐
│ FastText        │
│ Training         │
│                  │
│ - model:         │
│   skipgram       │
│ - dim: 100       │
│ - minCount: 1    │
│ - epoch: 10      │
│ - lr: 0.05       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Word Embeddings   │
│ (100 dimensions)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Sentence         │
│ Embeddings       │
│ (Average word    │
│  vectors)        │
└──────────────────┘
```

#### Design Choices

**dim = 100**:
- **Rationale**: Balances representational capacity and computational efficiency
- **Trade-off**: Lower dimensions are faster but may lose information; higher dimensions are more expressive but slower

**model = 'skipgram'**:
- **Rationale**: Skip-gram typically performs better than CBOW for rare words
- **Benefit**: Better for code-mixed text with many low-frequency words

**minCount = 1**:
- **Rationale**: Preserves all words, important for code-mixed text with many unique terms
- **Trade-off**: May include noise but captures rare but important words

**Training on All Data**:
- **Rationale**: Better embeddings when trained on the full corpus
- **Benefit**: Captures domain-specific patterns in code-mixed sarcasm

**Sentence Embedding = Average**:
- **Rationale**: Simple, effective method for fixed-length sentence representation
- **Alternative Considered**: Weighted average, but simple average works well

### 3.3 Sequence Tokenization (BiLSTM)

#### Algorithm

**Tokenization**:
- Maps words to integer IDs based on frequency
- Most frequent words get lower IDs
- Out-of-vocabulary words mapped to special token `<OOV>`

**Padding**:
- Sequences padded/truncated to fixed length
- Padding: 'post' (add zeros at end)
- Truncation: 'post' (remove from end)

#### Architecture

```
Text
    │
    ▼
┌──────────────────┐
│ Keras Tokenizer  │
│                  │
│ - vocab_size:    │
│   20000          │
│ - oov_token:     │
│   <OOV>          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Integer Sequences │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Padding/         │
│ Truncation       │
│                  │
│ - max_len:       │
│   95th           │
│   percentile     │
│ - padding: post  │
│ - truncating:    │
│   post           │
└──────────────────┘
```

#### Design Choices

**vocab_size = 20000**:
- **Rationale**: Captures most frequent words while managing memory
- **Trade-off**: Larger vocab captures more words but increases model size

**max_len = 95th percentile**:
- **Rationale**: Captures most sequences while avoiding extreme outliers
- **Benefit**: Adaptive to data distribution, avoids excessive padding

**padding = 'post'**:
- **Rationale**: Standard practice; padding at end doesn't affect LSTM processing
- **Alternative**: 'pre' padding also common, but 'post' is simpler

### 3.4 Transformer Tokenization

#### Algorithm

**Subword Tokenization**:
- Uses model-specific tokenizers (BPE, SentencePiece, etc.)
- Splits words into subwords to handle OOV words
- Example: "unhappiness" → ["un", "##happiness"]

**Special Tokens**:
- `[CLS]`: Classification token (for BERT)
- `[SEP]`: Separator token
- `[PAD]`: Padding token
- `[UNK]`: Unknown token

#### Architecture

```
Text
    │
    ▼
┌──────────────────┐
│ Model-Specific   │
│ Tokenizer        │
│                  │
│ Models:          │
│ - XLM-RoBERTa    │
│ - mBERT          │
│ - Indic-BERT     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Token IDs +      │
│ Attention Mask   │
│                  │
│ - max_length: 96 │
│ - truncation:    │
│   True           │
│ - padding:       │
│   max_length     │
└──────────────────┘
```

#### Design Choices

**max_length = 96**:
- **Rationale**: Most tweets fit within this length; balances context and efficiency
- **Trade-off**: Longer sequences provide more context but increase computation

**Model Selection**:
- **XLM-RoBERTa**: Strong multilingual performance, large vocabulary
- **mBERT**: Specifically trained for multilingual tasks
- **Indic-BERT**: Optimized for Indian languages, better for code-mixed text

## 4. Classification Models

### 4.1 Random Forest

#### Algorithm

Random Forest is an ensemble method that combines multiple decision trees.

**Training Process**:
1. **Bootstrap Sampling**: Create multiple training sets by sampling with replacement
2. **Feature Subsampling**: For each tree, randomly select a subset of features
3. **Tree Construction**: Build decision trees using Gini impurity or entropy
4. **Voting**: For prediction, aggregate predictions from all trees (majority vote for classification)

**Gini Impurity**:
```
Gini(D) = 1 - Σ(p_i)^2
```
where p_i is the proportion of samples belonging to class i.

**Information Gain**:
```
IG(D, A) = Entropy(D) - Σ(|D_v|/|D|) × Entropy(D_v)
```

#### Architecture

```
Training Data
    │
    ▼
┌──────────────────┐
│ Bootstrap        │
│ Sampling         │
│ (n_estimators    │
│  times)          │
└────────┬─────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Decision     │ │ Decision     │ │ Decision     │ │ Decision     │
│ Tree 1       │ │ Tree 2       │ │ Tree 3       │ │ Tree N       │
│              │ │              │ │              │ │              │
│ Features:    │ │ Features:    │ │ Features:    │ │ Features:    │
│ Random subset│ │ Random subset│ │ Random subset│ │ Random subset│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       └────────────────┼────────────────┼────────────────┘
                        │
                        ▼
                ┌──────────────┐
                │ Majority     │
                │ Vote         │
                └──────────────┘
```

#### Design Choices

**n_estimators = 100**:
- **Rationale**: Good balance between performance and training time
- **Trade-off**: More trees improve performance but increase training time

**max_depth = None**:
- **Rationale**: Allows trees to grow fully, capturing complex patterns
- **Trade-off**: May overfit, but ensemble averaging mitigates this

**class_weight = 'balanced'**:
- **Rationale**: Handles slight class imbalance (53.6% vs 46.4%)
- **Benefit**: Prevents bias toward majority class

**n_jobs = -1**:
- **Rationale**: Parallelizes tree construction across all CPU cores
- **Benefit**: Significantly faster training

### 4.2 Support Vector Machine (SVM)

#### Algorithm

SVM finds the optimal hyperplane that separates classes with maximum margin.

**Objective Function**:
```
Minimize: (1/2)||w||² + C × Σξ_i
Subject to: y_i(w·x_i + b) ≥ 1 - ξ_i
```

Where:
- `w`: Weight vector
- `C`: Regularization parameter
- `ξ_i`: Slack variables (allow misclassification)
- `b`: Bias term

**Dual Formulation** (for large datasets):
```
Maximize: Σα_i - (1/2)ΣΣα_iα_j y_i y_j K(x_i, x_j)
Subject to: 0 ≤ α_i ≤ C, Σα_i y_i = 0
```

**Linear Kernel**:
```
K(x_i, x_j) = x_i · x_j
```

#### Architecture

```
Training Data
    │
    ▼
┌──────────────────┐
│ Feature Vectors  │
│ (TF-IDF or        │
│  Embeddings)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Find Optimal     │
│ Hyperplane       │
│                  │
│ - C: 1.0         │
│ - Kernel: Linear │
│ - dual: False    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Support Vectors  │
│ (Decision        │
│  Boundary)       │
└──────────────────┘
```

#### Design Choices

**C = 1.0**:
- **Rationale**: Default value; balances margin width and misclassification penalty
- **Trade-off**: Higher C = stricter margin (less tolerance for errors), lower C = softer margin

**dual = False**:
- **Rationale**: For large datasets (n_samples > n_features), primal formulation is faster
- **Benefit**: More efficient for sparse TF-IDF matrices

**max_iter = 1000**:
- **Rationale**: Sufficient iterations for convergence on this dataset size
- **Trade-off**: More iterations ensure convergence but take longer

**class_weight = 'balanced'**:
- **Rationale**: Handles class imbalance
- **Benefit**: Prevents bias toward majority class

**Kernel = Linear**:
- **Rationale**: Linear kernel is efficient and works well for high-dimensional sparse data
- **Alternative Considered**: RBF kernel, but linear performs well and is faster

### 4.3 Bidirectional LSTM (BiLSTM)

#### Algorithm

**LSTM (Long Short-Term Memory)**:
LSTM addresses the vanishing gradient problem in RNNs using gating mechanisms.

**LSTM Cell Equations**:
```
Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate Values: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State: C_t = f_t * C_{t-1} + i_t * C̃_t
Output Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State: h_t = o_t * tanh(C_t)
```

**Bidirectional LSTM**:
- Processes sequence in both forward and backward directions
- Concatenates outputs: `[h_forward, h_backward]`

#### Architecture

```
Input: Padded Sequences (n_samples × max_len)
    │
    ▼
┌─────────────────────────────────────┐
│ Embedding Layer                     │
│                                     │
│ Input: vocab_size                   │
│ Output: embedding_dim = 128        │
│ Shape: (batch, max_len, 128)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Bidirectional LSTM                  │
│                                     │
│ - Units: 64                         │
│ - return_sequences: True            │
│ - Output: (batch, max_len, 128)    │
│   (64 forward + 64 backward)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Global Max Pooling                   │
│                                     │
│ - Takes max over sequence dimension │
│ - Output: (batch, 128)              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Dropout (0.5)                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Dense Layer (64, ReLU)              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Dropout (0.5)                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Dense Layer (1, Sigmoid)            │
│                                     │
│ Output: Probability [0, 1]          │
└─────────────────────────────────────┘
```

#### Design Choices

**Embedding Dimension = 128**:
- **Rationale**: Good balance between representational capacity and model size
- **Trade-off**: Larger embeddings capture more information but increase parameters

**LSTM Units = 64**:
- **Rationale**: Sufficient capacity for sequence modeling without overfitting
- **Trade-off**: More units = more capacity but more parameters

**Bidirectional**:
- **Rationale**: Captures context from both directions, important for sarcasm
- **Benefit**: Better understanding of sentence structure and context

**return_sequences = True**:
- **Rationale**: Returns full sequence output for pooling layer
- **Benefit**: Allows pooling to select most important features

**Global Max Pooling**:
- **Rationale**: Captures most salient features across the sequence
- **Alternative Considered**: Average pooling, but max pooling often works better for classification

**Dropout = 0.5**:
- **Rationale**: Standard dropout rate for regularization
- **Benefit**: Prevents overfitting by randomly zeroing 50% of activations

**Dense Layer (64, ReLU)**:
- **Rationale**: Non-linear transformation before final classification
- **Benefit**: Adds model capacity for complex decision boundaries

**Optimizer = Adam (lr = 3e-4)**:
- **Rationale**: Adam adapts learning rate per parameter, works well for RNNs
- **Learning Rate**: 3e-4 is a good default for Adam

**Loss = Binary Crossentropy**:
- **Rationale**: Standard loss for binary classification
- **Formula**: `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`

**Early Stopping (patience = 2)**:
- **Rationale**: Prevents overfitting by stopping when validation loss stops improving
- **Benefit**: Automatically finds optimal stopping point

### 4.4 Transformer Models

#### Algorithm

**Transformer Architecture**:
Transformers use self-attention mechanisms to process sequences.

**Self-Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d_k: Dimension of keys

**Multi-Head Attention**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Transformer Block**:
1. Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Feed-Forward Network
4. Residual Connection + Layer Normalization

**Fine-tuning**:
- Pre-trained transformer + classification head
- Classification head: Linear layer mapping hidden states to 2 classes

#### Architecture

```
Input: Tokenized Text
    │
    ▼
┌─────────────────────────────────────┐
│ Pre-trained Transformer Base        │
│                                     │
│ Models:                             │
│ - XLM-RoBERTa (125M params)        │
│ - mBERT (110M params)               │
│ - Indic-BERT (110M params)         │
│                                     │
│ Architecture:                       │
│ - Embedding Layer                   │
│ - Transformer Blocks (12 layers)   │
│   - Multi-Head Attention            │
│   - Feed-Forward                    │
│   - Layer Norm                      │
│ - Pooler (for [CLS] token)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Classification Head                 │
│                                     │
│ - Dropout                           │
│ - Linear (hidden_size → 2)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Output: Logits [2]                   │
│ → Softmax → Probabilities            │
└─────────────────────────────────────┘
```

#### Design Choices

**Model Selection**:

**XLM-RoBERTa**:
- **Rationale**: Strong multilingual performance, trained on 100 languages
- **Benefit**: Handles code-mixed text well due to multilingual training
- **Architecture**: 12 layers, 768 hidden size, 12 attention heads

**mBERT (Multilingual BERT)**:
- **Rationale**: Specifically designed for multilingual tasks
- **Benefit**: Trained on 104 languages including Hindi
- **Architecture**: 12 layers, 768 hidden size, 12 attention heads

**Indic-BERT**:
- **Rationale**: Optimized for Indian languages
- **Benefit**: Better understanding of Hindi-English code-mixing patterns
- **Architecture**: Similar to BERT but trained on Indic languages

**Fine-tuning Strategy**:

**Learning Rate = 2e-5**:
- **Rationale**: Standard learning rate for transformer fine-tuning
- **Trade-off**: Too high = unstable training, too low = slow convergence

**Batch Size = 8 (train), 16 (eval)**:
- **Rationale**: Small batch size for CPU training
- **Trade-off**: Larger batches are more stable but require more memory

**Epochs = 3**:
- **Rationale**: Transformers converge quickly with fine-tuning
- **Trade-off**: More epochs may improve performance but risk overfitting

**Weight Decay = 0.01**:
- **Rationale**: L2 regularization to prevent overfitting
- **Benefit**: Helps generalization

**Evaluation Strategy = 'epoch'**:
- **Rationale**: Evaluate after each epoch to monitor progress
- **Benefit**: Early detection of overfitting

**Metric for Best Model = 'f1'**:
- **Rationale**: F1 score balances precision and recall
- **Benefit**: Better metric for imbalanced datasets than accuracy

**Max Length = 96**:
- **Rationale**: Most tweets fit within this length
- **Trade-off**: Longer sequences provide more context but increase computation

## 5. Design Choice Rationale

### 5.1 Why Multiple Approaches?

**Comprehensive Comparison**:
- Different approaches capture different aspects of sarcasm
- Systematic comparison identifies best method for this specific task

**Complementary Strengths**:
- **TF-IDF**: Captures lexical patterns and n-grams
- **FastText**: Handles OOV words and morphological variations
- **BiLSTM**: Models sequential dependencies
- **Transformers**: Captures complex contextual relationships

### 5.2 Feature Extraction Choices

**TF-IDF for Traditional ML**:
- **Rationale**: Proven effective for text classification, interpretable
- **Benefit**: Fast, works well with linear models

**FastText for Embeddings**:
- **Rationale**: Handles code-mixed text well, subword information
- **Benefit**: Better for OOV words common in transliterated Hindi

**Sequence Tokenization for BiLSTM**:
- **Rationale**: Preserves word order, necessary for sequential models
- **Benefit**: Captures sequential patterns in sarcasm

**Transformer Tokenization**:
- **Rationale**: Pre-trained tokenizers optimized for specific models
- **Benefit**: Leverages pre-trained knowledge

### 5.3 Classifier Choices

**Random Forest**:
- **Rationale**: Robust, handles non-linear relationships, feature importance
- **Benefit**: Good performance, interpretable feature importance

**SVM**:
- **Rationale**: Effective for high-dimensional sparse data
- **Benefit**: Strong generalization, works well with TF-IDF

**BiLSTM**:
- **Rationale**: Models sequential dependencies important for sarcasm
- **Benefit**: Captures long-range dependencies

**Transformers**:
- **Rationale**: State-of-the-art for NLP, pre-trained on large corpora
- **Benefit**: Leverages transfer learning, strong contextual understanding

### 5.4 Hyperparameter Choices

**Common Patterns**:
- **Class Weight = 'balanced'**: Handles slight class imbalance
- **Random Seed = 42**: Ensures reproducibility
- **Stratified Splits**: Maintains class distribution
- **Early Stopping**: Prevents overfitting

**Model-Specific**:
- **TF-IDF**: Focus on vocabulary size and n-gram range
- **FastText**: Balance embedding dimension and training epochs
- **BiLSTM**: Balance model capacity and regularization
- **Transformers**: Focus on learning rate and fine-tuning epochs

## 6. Model Comparison Summary

| Model | Architecture | Key Features | Strengths | Limitations |
|-------|-------------|--------------|-----------|-------------|
| TF-IDF + RF | Sparse features + Ensemble trees | Lexical patterns, n-grams | Fast, interpretable | May miss semantic relationships |
| TF-IDF + SVM | Sparse features + Linear classifier | High-dimensional separation | Strong generalization | Linear decision boundary |
| FastText + RF | Dense embeddings + Ensemble | Subword information, OOV handling | Handles code-mixing well | Fixed-length sentence representation |
| FastText + SVM | Dense embeddings + Linear | Efficient embeddings | Fast training | Linear separation |
| BiLSTM | Sequential neural network | Sequential dependencies | Captures context | Requires more data, slower |
| Transformers | Pre-trained + Fine-tuning | Contextual understanding | State-of-the-art potential | Requires GPU for training, slower |

## 7. Best Model: FastText + Random Forest

**Performance**: 97.72% accuracy, 97.71% F1 score

**Why It Works Best**:
1. **FastText Embeddings**: 
   - Handles code-mixed text effectively
   - Subword information captures transliteration variations
   - OOV word handling important for informal text

2. **Random Forest**:
   - Non-linear decision boundaries
   - Robust to noise
   - Ensemble averaging reduces overfitting

3. **Combination**:
   - Dense embeddings provide semantic information
   - Random Forest captures complex patterns
   - Good balance of performance and efficiency

## 8. Conclusion

The SarcasmLens model framework employs a diverse set of architectures and algorithms, each with specific design choices optimized for code-mixed sarcasm detection. The systematic comparison reveals that FastText embeddings combined with Random Forest provides the best performance, achieving 97.72% accuracy. This success stems from FastText's ability to handle code-mixed text and OOV words, combined with Random Forest's robust non-linear classification capabilities.

The framework demonstrates that:
- **Feature extraction matters**: FastText embeddings outperform TF-IDF for this task
- **Model choice matters**: Random Forest outperforms SVM for this dataset
- **Deep learning has potential**: BiLSTM and Transformers show promise but may need more data or different configurations
- **Design choices are critical**: Hyperparameters and preprocessing significantly impact performance

---

*This document describes the complete model framework, including architectures, algorithms, and design choices for the SarcasmLens project.*


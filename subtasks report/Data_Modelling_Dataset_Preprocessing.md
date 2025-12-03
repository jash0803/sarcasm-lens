# Data Modelling - Dataset + Preprocessing

## 1. Overview

This document describes the data collection, dataset composition, preprocessing pipeline, and data modeling approach for the SarcasmLens project. The project combines multiple datasets from different sources to create a comprehensive corpus of Hindi-English code-mixed sarcastic and non-sarcastic social media text.

## 2. Dataset Sources

The project utilizes four primary datasets, each contributing unique characteristics to the combined corpus:

### 2.1 Swami et al. Dataset

**Source**: Swami, S., Khandelwal, A., Singh, V., Akhtar, S. S., & Shrivastava, M. (2018).  
*A Corpus of English-Hindi Code-Mixed Tweets for Sarcasm Detection*.  
Available at: [https://arxiv.org/abs/1805.11869](https://arxiv.org/abs/1805.11869)

**Location**: `datasets/sahil swami/`

**Format**:
- `Sarcasm_tweets.txt`: Contains tweet IDs and corresponding text
- `Sarcasm_tweet_truth.txt`: Contains tweet IDs and labels (YES/NO for sarcasm)

**Processing**:
- The `create_csv.py` script combines tweet IDs with their labels and text
- Labels are converted from YES/NO to binary (1/0)
- Output: `sarcasm_dataset.csv` with columns: `tweet_id`, `label`, `text`

**Characteristics**:
- Focus on English-Hindi code-mixed tweets
- Manually annotated for sarcasm
- Contains both sarcastic and non-sarcastic examples

### 2.2 HackArena Dataset

**Source**: Kaggle Dataset - *HackArena Theme 2 – Multilingual Sarcasm Detection*  
Available at: [https://www.kaggle.com/datasets/divyanshu134/hackarena-theme-2-multilingual-sarcasm-detection/data](https://www.kaggle.com/datasets/divyanshu134/hackarena-theme-2-multilingual-sarcasm-detection/data)

**Location**: `datasets/hackarena/`

**Format**:
- Pre-split into `train.csv` and `test.csv`
- Contains multilingual sarcasm detection data

**Processing**:
- Labels converted from YES/NO to binary (1/0) using `convert_labels.py`
- Preprocessed versions: `hackarena_train_processed.csv`, `hackarena_test_processed.csv`

**Characteristics**:
- Multilingual focus (includes Hindi-English code-mixed text)
- Competition dataset with train/test splits
- Diverse domains and topics

### 2.3 Mendeley Dataset

**Source**: Research dataset from Mendeley data repository

**Location**: `datasets/mendeley data/`

**Format**:
- Contains code-mixed text with sentiment labels
- Original file: `labeled_sentiment_output_codemixed.csv`

**Processing**:
- `extract_text_and_sentiment.py` extracts text and converts sentiment to binary labels
- Positive sentiment mapped to label 1 (sarcastic), others to 0
- Output: `mendeley_data_text_label.csv`

**Characteristics**:
- Code-mixed social media content
- Sentiment-based labeling approach
- Includes various text styles and formats

### 2.4 Aggarwal et al. Dataset

**Source**: Aggarwal, A., Wadhawan, A., Chaudhary, A., & Maurya, K. (2020).  
*"Did you really mean what you said?" : Sarcasm Detection in Hindi-English Code-Mixed Data using Bilingual Word Embeddings*.  
Available at: [https://arxiv.org/abs/2010.00310](https://arxiv.org/abs/2010.00310)

**Location**: `datasets/akshita agrawall/`

**Format**:
- Contains tweet IDs and sarcasm labels
- `Sarcasm_dataset.csv` with tweet IDs and labels

**Processing**:
- `fetch_tweets.py` can fetch full tweet text from X (Twitter) API using tweet IDs
- Preprocessed version: `akshita_processed.csv`

**Characteristics**:
- Focus on bilingual word embeddings for code-mixed text
- Research-oriented dataset with detailed annotations

## 3. Dataset Combination

### 3.1 Combination Strategy

The `combine_datasets.py` script merges all individual datasets into a unified corpus:

**Process**:
1. **Input**: Multiple CSV files from different sources
2. **Extraction**: Extracts only `text` and `label` columns from each dataset
3. **Validation**: Checks for required columns and handles missing values
4. **Cleaning**: Removes rows with missing or empty text
5. **Merging**: Concatenates all datasets into a single DataFrame
6. **Output**: `datasets/combined_dataset.csv`

**Key Features**:
- Handles different column names across datasets
- Removes duplicates and invalid entries
- Preserves label consistency (binary: 0 = non-sarcastic, 1 = sarcastic)
- Reports statistics for each dataset and the combined result

### 3.2 Combined Dataset Statistics

**Total Samples**: 15,099

**Label Distribution**:
- **Non-sarcastic (0)**: 8,091 samples (53.6%)
- **Sarcastic (1)**: 7,008 samples (46.4%)

**Text Characteristics**:
- **Mean length**: 80.89 characters
- **Median length**: 76 characters
- **Standard deviation**: 33.62 characters
- **Min length**: 6 characters
- **Max length**: 175 characters
- **25th percentile**: 53 characters
- **75th percentile**: 112 characters

**Class Balance**: The dataset is relatively balanced with a slight bias toward non-sarcastic examples, which is typical for sarcasm detection tasks.

## 4. Data Preprocessing Pipeline

### 4.1 Preprocessing Steps

The `preprocess_data.py` script implements a comprehensive text preprocessing pipeline designed specifically for code-mixed social media text:

#### Step 1: Text Normalization
```python
text = str(text).lower()  # Convert to lowercase
```
- Normalizes case to ensure consistent token matching
- Important for code-mixed text where capitalization may be inconsistent

#### Step 2: Remove User Mentions
```python
text = re.sub(r'@\w+', '', text)  # Remove @ mentions
```
- Removes Twitter/X user mentions (e.g., `@username`, `@TripleTalaq`)
- These are typically not relevant for sarcasm detection

#### Step 3: Hashtag Processing
```python
# Split camelCase hashtags
hashtags = re.findall(r'#\w+', text)
for hashtag in hashtags:
    hashtag_text = hashtag[1:]  # Remove #
    split_hashtag = split_camel_case(hashtag_text)
    text = text.replace(hashtag, split_hashtag)
```
- Identifies hashtags and splits camelCase words
- Example: `#ILoveIndia` → `I Love India`
- Preserves semantic content while making hashtags more readable

#### Step 4: URL Removal
```python
text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
```
- Removes HTTP/HTTPS URLs and www links
- URLs typically don't contribute to sarcasm detection

#### Step 5: Punctuation Removal
```python
text = text.translate(str.maketrans('', '', string.punctuation))
```
- Removes all punctuation marks
- **Note**: This is a design choice - punctuation can sometimes signal sarcasm (e.g., excessive exclamation marks), but removing it simplifies the feature space

#### Step 6: Stop Word Removal
```python
STOP_WORDS = set(stopwords.words('english'))
words = text.split()
words = [word for word in words if word.lower() not in STOP_WORDS]
text = ' '.join(words)
```
- Removes English stop words using NLTK
- **Important**: Only English stop words are removed; Hindi words are preserved
- This helps focus on content words while maintaining code-mixed structure

#### Step 7: Whitespace Normalization
```python
text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
text = text.strip()  # Remove leading/trailing spaces
```
- Normalizes multiple spaces to single spaces
- Removes leading/trailing whitespace

### 4.2 Preprocessing Design Decisions

**Rationale for Each Step**:

1. **Lowercasing**: Ensures consistent tokenization, especially important for code-mixed text with inconsistent capitalization.

2. **Mention Removal**: User mentions are typically not informative for sarcasm detection and can introduce noise.

3. **Hashtag Processing**: Splitting camelCase hashtags preserves semantic information that might be lost if hashtags were simply removed.

4. **URL Removal**: URLs don't contribute to sarcasm detection and can introduce noise.

5. **Punctuation Removal**: While punctuation can sometimes signal sarcasm, removing it:
   - Simplifies the feature space
   - Reduces vocabulary size
   - Makes the model focus on lexical and semantic patterns
   - **Trade-off**: May lose some sarcasm signals (e.g., excessive punctuation)

6. **Stop Word Removal**: 
   - Reduces dimensionality
   - Focuses on content words
   - **Important**: Only English stop words removed; Hindi words preserved to maintain code-mixed structure

7. **Whitespace Normalization**: Ensures consistent formatting for downstream processing.

### 4.3 Preprocessing Example

**Before Preprocessing**:
```
@username lol bc badaa gaandu hai jaa lodu #SarcasmAlert https://example.com
```

**After Preprocessing**:
```
lol bc badaa gaandu hai jaa lodu sarcasm alert
```

**Changes**:
- `@username` removed
- `#SarcasmAlert` → `sarcasm alert` (camelCase split)
- URL removed
- Converted to lowercase
- Punctuation removed
- English stop words removed (if any)
- Whitespace normalized

## 5. Data Modeling Approach

### 5.1 Data Structure

The final combined dataset has a simple, clean structure:

**Schema**:
```
text: str    # Preprocessed text (code-mixed Hindi-English)
label: int   # Binary label (0 = non-sarcastic, 1 = sarcastic)
```

**File Format**: CSV (UTF-8 encoding)

**Sample Data**:
```csv
text,label
lol bc badaa gaandu hai jaa lodu tere se kuch nahi hoga,1
ek aadmi pareshan ho kar ae bhagwan aisi zindagi se maut achi,1
ussase jayada dukhad congress ka nichi koti ki politics,0
```

### 5.2 Label Encoding

**Binary Classification**:
- **0**: Non-sarcastic (literal statements, genuine sentiment)
- **1**: Sarcastic (ironic, sarcastic expressions)

**Label Consistency**:
- All datasets are normalized to use the same binary encoding
- Original labels (YES/NO, Positive/Negative, etc.) are converted to 0/1
- Ensures consistency across different data sources

### 5.3 Train-Validation-Test Split

**Standard Split Strategy**:
- **Training Set**: 80% of data
- **Test Set**: 20% of data
- **Validation Set**: Created from training set (typically 10-20% of training data)

**Stratification**:
- All splits use stratified sampling to maintain class distribution
- Ensures both training and test sets have similar proportions of sarcastic/non-sarcastic examples
- Prevents class imbalance issues in evaluation

**Random Seed**: Fixed seed (42) for reproducibility

**Implementation** (example from training scripts):
```python
from sklearn.model_selection import train_test_split

# Initial split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)

# Further split training into train/validation if needed
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,  # 10% of training = validation
    random_state=42,
    stratify=y_train
)
```

### 5.4 Data Quality Considerations

**Handling Missing Data**:
- Rows with missing `text` or `label` values are removed during combination
- Empty text after preprocessing is removed

**Handling Duplicates**:
- The combination script doesn't explicitly remove duplicates
- This is a potential area for future improvement

**Text Length Filtering**:
- No explicit minimum/maximum length filtering applied
- Model architectures handle variable-length sequences through padding/truncation

**Encoding Issues**:
- All files use UTF-8 encoding to handle Hindi characters in Roman script
- Special characters and emojis are preserved (though emojis may be removed during tokenization depending on the model)

## 6. Dataset Characteristics

### 6.1 Code-Mixing Patterns

The dataset exhibits various code-mixing patterns:

**Intra-sentential Mixing** (within sentence):
- "kya aapko lagta hai ki yeh **brilliant** idea hai?"
- "lol bc badaa gaandu hai jaa lodu"

**Inter-sentential Mixing** (between sentences):
- Mixing Hindi and English sentences within the same post

**Morphological Mixing**:
- Combining Hindi and English morphemes
- Creative word formations

**Transliteration Variations**:
- Hindi words written in Roman script with inconsistent spelling
- Example: "kya", "kiya", "kiyaa" (all meaning "what/did")

### 6.2 Domain Diversity

The combined dataset covers multiple domains:
- **Politics**: Political commentary and satire
- **Entertainment**: Movie, TV, and celebrity-related sarcasm
- **Daily Life**: Everyday situations and observations
- **Social Issues**: Commentary on social and cultural topics
- **Technology**: Tech-related sarcastic comments

### 6.3 Sarcasm Indicators

Common patterns observed in sarcastic examples:
- **Lexical Markers**: "lol", "haha", "sure", "obviously"
- **Exaggeration**: Over-the-top statements
- **Rhetorical Questions**: Questions not meant to be answered
- **Contrast**: Statements that contrast with expected sentiment
- **Cultural References**: References to Indian culture, politics, or popular culture

## 7. Data Pipeline Summary

### 7.1 Complete Pipeline

```
Raw Datasets (Multiple Sources)
    ↓
[create_csv.py] - Convert raw formats to CSV
    ↓
Individual CSV Files
    ↓
[preprocess_data.py] - Text preprocessing
    ↓
Preprocessed CSV Files
    ↓
[convert_labels.py] - Normalize labels (if needed)
    ↓
[extract_text_and_sentiment.py] - Extract relevant columns (if needed)
    ↓
Normalized Individual Datasets
    ↓
[combine_datasets.py] - Merge all datasets
    ↓
combined_dataset.csv (15,099 samples)
    ↓
Model Training Scripts
    ↓
Train/Test Split (80/20)
    ↓
Feature Extraction & Model Training
```

### 7.2 Key Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `create_csv.py` | Convert raw text files to CSV | `.txt` files | `sarcasm_dataset.csv` |
| `preprocess_data.py` | Clean and normalize text | Raw CSV | Preprocessed CSV |
| `convert_labels.py` | Normalize label formats | CSV with YES/NO | CSV with 0/1 |
| `extract_text_and_sentiment.py` | Extract text and labels | Full dataset | Text+Label CSV |
| `combine_datasets.py` | Merge multiple datasets | Multiple CSVs | `combined_dataset.csv` |
| `fetch_tweets.py` | Fetch tweet text from API | Tweet IDs | Full tweet text |

## 8. Challenges and Solutions

### 8.1 Challenges Encountered

1. **Dataset Format Inconsistency**:
   - Different datasets use different formats (TXT, CSV, different column names)
   - **Solution**: Created conversion scripts for each dataset format

2. **Label Encoding Variations**:
   - Some datasets use YES/NO, others use 1/0, others use sentiment labels
   - **Solution**: Normalized all labels to binary 0/1 encoding

3. **Code-Mixing Complexity**:
   - Inconsistent transliteration, language switching patterns
   - **Solution**: Preprocessing pipeline that preserves code-mixed structure while normalizing text

4. **Class Imbalance**:
   - Slight imbalance in the combined dataset
   - **Solution**: Used stratified sampling and class weights in models

5. **Noisy Text**:
   - Social media text is inherently noisy
   - **Solution**: Comprehensive preprocessing pipeline to clean text while preserving sarcasm signals

### 8.2 Future Improvements

1. **Deduplication**: Add explicit duplicate removal in the combination script
2. **Length Filtering**: Consider filtering extremely short or long texts
3. **Emoji Handling**: Explicit handling of emojis (preserve or remove based on analysis)
4. **Punctuation Analysis**: Evaluate whether preserving certain punctuation (e.g., exclamation marks) improves performance
5. **Language Tagging**: Add explicit language tags to help models understand code-mixing patterns
6. **Data Augmentation**: Consider data augmentation techniques for code-mixed text

## 9. Data Statistics Summary

| Metric | Value |
|--------|-------|
| Total Samples | 15,099 |
| Non-Sarcastic (0) | 8,091 (53.6%) |
| Sarcastic (1) | 7,008 (46.4%) |
| Mean Text Length | 80.89 characters |
| Median Text Length | 76 characters |
| Min Text Length | 6 characters |
| Max Text Length | 175 characters |
| Number of Datasets Combined | 4 |
| Preprocessing Steps | 7 |
| Final Format | CSV (text, label) |

## 10. Conclusion

The data modeling and preprocessing pipeline for SarcasmLens successfully combines multiple datasets from different sources into a unified, clean corpus suitable for training sarcasm detection models. The preprocessing pipeline is specifically designed to handle the unique challenges of code-mixed social media text while preserving the linguistic and stylistic information necessary for accurate sarcasm detection.

The combined dataset of 15,099 samples provides a solid foundation for training and evaluating various machine learning and deep learning models, as demonstrated by the high performance achieved (97.72% accuracy with FastText + Random Forest).

---

*This document describes the complete data modeling and preprocessing pipeline for the SarcasmLens project, covering dataset sources, preprocessing steps, data characteristics, and modeling approach.*


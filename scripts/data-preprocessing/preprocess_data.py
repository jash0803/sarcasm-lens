import pandas as pd
import re
import csv
import string
import nltk
from nltk.corpus import stopwords
import fire

# Download stopwords if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Get English stop words
STOP_WORDS = set(stopwords.words('english'))

def split_camel_case(text):
    """Split camelCase hashtags into separate words.
    Example: #ILoveIndia -> I Love India
    """
    # Insert space before uppercase letters that follow lowercase letters or numbers
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    return text

def preprocess_text(text):
    """Preprocess text by removing @ mentions and splitting camelCase hashtags."""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove @ mentions (e.g., @username, @TripleTalaq)
    text = re.sub(r'@\w+', '', text)
    
    # Find all hashtags and process them
    hashtags = re.findall(r'#\w+', text)
    
    for hashtag in hashtags:
        # Remove the # symbol and split camelCase
        hashtag_text = hashtag[1:]  # Remove #
        split_hashtag = split_camel_case(hashtag_text)
        # Replace the original hashtag with the split version
        text = text.replace(hashtag, split_hashtag)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stop words
    words = text.split()
    words = [word for word in words if word.lower() not in STOP_WORDS]
    text = ' '.join(words)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def main(
    input_file: str = "datasets/sahil swami/sarcasm_dataset.csv",
    output_file: str = "datasets/sahil swami/sarcasm_dataset_processed.csv",
    text_column: str = "text"
):
    """
    Preprocess a dataset and write the cleaned text back to CSV.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path where the cleaned CSV should be stored.
        text_column: Name of the column containing raw text.
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in dataset columns: {df.columns.tolist()}"
        )

    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    # Capture sample before preprocessing for reference
    original_sample = df[text_column].head(3).tolist()

    # Preprocess the text column
    print(f"Preprocessing text column '{text_column}'...")
    df[text_column] = df[text_column].apply(preprocess_text)

    # Remove rows with empty text after preprocessing
    non_empty_mask = df[text_column].str.strip() != ""
    df_preprocessed = df[non_empty_mask].copy()

    print(f"Preprocessed dataset shape: {df_preprocessed.shape}")

    # Save preprocessed dataset
    print(f"Saving preprocessed dataset to {output_file}...")
    df_preprocessed.to_csv(output_file, index=False, encoding='utf-8')

    print("Preprocessing complete!")
    print("\nSample of original text:")
    print(original_sample)
    print("\nSample of preprocessed text:")
    print(df_preprocessed[text_column].head(3).tolist())

    # Show label distribution if available
    if "label" in df_preprocessed.columns:
        print("\nLabel distribution:")
        print(df_preprocessed["label"].value_counts())
    else:
        print("\nLabel column not found; skipping label distribution.")

if __name__ == "__main__":
    fire.Fire(main)


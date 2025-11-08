import pandas as pd
import re
import csv

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
    
    text = str(text)
    
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
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def main():
    # Read the original dataset
    input_file = "datasets/sahil swami dataset/sarcasm_dataset.csv"
    output_file = "datasets/sahil swami dataset/sarcasm_dataset_preprocessed.csv"
    
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Preprocess the text column
    print("Preprocessing text...")
    df['text_preprocessed'] = df['text'].apply(preprocess_text)
    
    # Create a new dataframe with preprocessed text
    df_preprocessed = df[['tweet_id', 'label', 'text_preprocessed']].copy()
    df_preprocessed.rename(columns={'text_preprocessed': 'text'}, inplace=True)
    
    # Remove rows with empty text after preprocessing
    df_preprocessed = df_preprocessed[df_preprocessed['text'].str.strip() != '']
    
    print(f"Preprocessed dataset shape: {df_preprocessed.shape}")
    
    # Save preprocessed dataset
    print(f"Saving preprocessed dataset to {output_file}...")
    df_preprocessed.to_csv(output_file, index=False, encoding='utf-8')
    
    print("Preprocessing complete!")
    print(f"\nSample of original text:")
    print(df['text'].head(3).tolist())
    print(f"\nSample of preprocessed text:")
    print(df_preprocessed['text'].head(3).tolist())
    
    # Show label distribution
    print(f"\nLabel distribution:")
    print(df_preprocessed['label'].value_counts())

if __name__ == "__main__":
    main()


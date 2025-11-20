import pandas as pd
import fire


def main(
    input_file: str = "datasets/mendeley data/mendeley_data_processed.csv",
    output_file: str = "datasets/mendeley data/mendeley_data_text_label.csv",
    text_column: str = "text",
    sentiment_column: str = "sentiment",
    positive_value: str = "Positive",
):
    """
    Extract the raw text column and convert sentiment to binary labels.

    Args:
        input_file: Path to the CSV file containing the full dataset.
        output_file: Destination CSV that will contain only text and label columns.
        text_column: Name of the column that contains the raw text to preserve.
        sentiment_column: Name of the column that contains sentiment labels.
        positive_value: Value in the sentiment column that should map to label 1.
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)

    missing_columns = [
        col
        for col in (text_column, sentiment_column)
        if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Dataset shape: {df.shape}")

    # Preserve the text exactly as stored
    texts = df[text_column].copy()

    # Map the sentiment column to binary labels
    sentiments = df[sentiment_column].astype(str).str.strip()
    labels = (sentiments == positive_value).astype(int)

    output_df = pd.DataFrame(
        {
            "text": texts,
            "label": labels,
        }
    )

    print(f"Saving extracted data to {output_file}...")
    output_df.to_csv(output_file, index=False, encoding="utf-8")

    print("Saved! Label distribution:")
    print(output_df["label"].value_counts())


if __name__ == "__main__":
    fire.Fire(main)



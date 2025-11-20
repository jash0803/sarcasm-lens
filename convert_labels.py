import pandas as pd
import fire


def main(
    input_file: str = "datasets/hackarena/hackarena_test_processed.csv",
    output_file: str = None,
    label_column: str = "label",
    yes_value: str = "YES",
    no_value: str = "NO",
    overwrite: bool = True,
):
    """
    Convert label column values from YES/NO to 1/0.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file. If None and overwrite=True, will overwrite input file.
        label_column: Name of the label column to convert (default: "label").
        yes_value: Value that should be mapped to 1 (default: "YES").
        no_value: Value that should be mapped to 0 (default: "NO").
        overwrite: If True and output_file is None, overwrites the input file.
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check if label column exists
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Show current label distribution
    print(f"\nCurrent label distribution:")
    print(df[label_column].value_counts())

    # Convert labels
    print(f"\nConverting labels: '{yes_value}' -> 1, '{no_value}' -> 0...")
    
    # Map YES to 1, NO to 0, keep other values as is (or you could raise error)
    label_mapping = {yes_value: 1, no_value: 0}
    
    # Check for unexpected values
    unique_labels = df[label_column].astype(str).str.strip().unique()
    unexpected = [val for val in unique_labels if val not in label_mapping]
    if unexpected:
        print(f"Warning: Found unexpected label values: {unexpected}")
        print("These will be left unchanged. Consider handling them explicitly.")
    
    # Convert labels
    df[label_column] = df[label_column].astype(str).str.strip().map(label_mapping).fillna(df[label_column])
    
    # Convert to int if all values are numeric
    try:
        df[label_column] = df[label_column].astype(int)
    except (ValueError, TypeError):
        print("Warning: Could not convert all labels to integers. Some values may remain unchanged.")

    # Show new label distribution
    print(f"\nNew label distribution:")
    print(df[label_column].value_counts().sort_index())

    # Determine output file
    if output_file is None:
        if overwrite:
            output_file = input_file
            print(f"\nOverwriting input file: {output_file}")
        else:
            # Create output filename by adding _converted suffix
            import os
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_converted{ext}"
            print(f"\nOutput file not specified. Saving to: {output_file}")

    # Save the converted dataset
    print(f"\nSaving converted dataset to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\nConversion complete! Saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)



import pandas as pd
import fire
from typing import List, Union
import os


def main(
    input_files: Union[str, List[str]],
    output_file: str,
    text_column: str = "text",
    label_column: str = "label",
):
    """
    Combine multiple datasets by extracting text and label columns from each file.

    Args:
        input_files: Comma-separated string of file paths or list of file paths.
        output_file: Path to the output CSV file that will contain combined data.
        text_column: Name of the text column to extract (default: "text").
        label_column: Name of the label column to extract (default: "label").
    """
    # Parse input files - handle both string (comma-separated) and list formats
    if isinstance(input_files, str):
        # Split by comma and strip whitespace
        file_list = [f.strip() for f in input_files.split(",")]
    else:
        file_list = input_files

    if not file_list:
        raise ValueError("No input files provided")

    print(f"Combining {len(file_list)} dataset(s)...")
    print(f"Input files: {file_list}")

    combined_dataframes = []

    for i, file_path in enumerate(file_list, 1):
        file_path = file_path.strip()
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}, skipping...")
            continue

        print(f"\n[{i}/{len(file_list)}] Processing: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")

            # Check if required columns exist
            missing_columns = []
            if text_column not in df.columns:
                missing_columns.append(text_column)
            if label_column not in df.columns:
                missing_columns.append(label_column)

            if missing_columns:
                print(f"  Warning: Missing columns {missing_columns}, skipping this file...")
                continue

            # Extract only text and label columns
            extracted_df = df[[text_column, label_column]].copy()
            
            # Remove rows with missing values
            initial_rows = len(extracted_df)
            extracted_df = extracted_df.dropna(subset=[text_column, label_column])
            final_rows = len(extracted_df)
            
            if initial_rows != final_rows:
                print(f"  Removed {initial_rows - final_rows} rows with missing values")
            
            # Remove rows with empty text
            extracted_df = extracted_df[extracted_df[text_column].astype(str).str.strip() != '']
            print(f"  Extracted {len(extracted_df)} rows")
            
            combined_dataframes.append(extracted_df)

        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}, skipping...")
            continue

    if not combined_dataframes:
        raise ValueError("No valid data was extracted from any input files")

    # Combine all dataframes
    print(f"\nCombining all datasets...")
    combined_df = pd.concat(combined_dataframes, ignore_index=True)
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Combined dataset columns: {combined_df.columns.tolist()}")
    
    # Show label distribution
    print(f"\nLabel distribution:")
    print(combined_df[label_column].value_counts().sort_index())
    
    # Save combined dataset
    print(f"\nSaving combined dataset to {output_file}...")
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nSuccessfully combined {len(combined_dataframes)} dataset(s) into {output_file}")
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    fire.Fire(main)



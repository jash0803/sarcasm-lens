#!/usr/bin/env python3
"""
Script to fetch sarcastic tweets from X API incrementally.
Fetches 50 tweets at a time and appends to the output CSV.
"""

import pandas as pd
import requests
import json
import os
from pathlib import Path
from typing import Set, List, Dict

# Configuration
BEARER_TOKEN = "YOUR_BEARER_TOKEN_HERE"  # Change this before running
INPUT_CSV = "datasets/akshita agrawall/Sarcasm_dataset.csv"
OUTPUT_CSV = "fetched_sarcastic_tweets.csv"
API_RESPONSES_DIR = "api_responses"
BATCH_SIZE = 50
API_BASE_URL = "https://api.x.com/2/tweets"

def load_already_fetched_tweet_ids(output_csv: str) -> Set[str]:
    """Load tweet IDs that have already been fetched."""
    if not os.path.exists(output_csv):
        return set()
    
    try:
        df = pd.read_csv(output_csv)
        return set(df['tweet_id'].astype(str))
    except Exception as e:
        print(f"Warning: Could not read existing output CSV: {e}")
        return set()

def get_sarcastic_tweet_ids(input_csv: str) -> List[str]:
    """Get all tweet IDs where sarcastic = 1."""
    df = pd.read_csv(input_csv)
    sarcastic_df = df[df['sarcastic'] == 1]
    return sarcastic_df['tweet_id'].astype(str).tolist()

def fetch_tweet_by_id(tweet_id: str, bearer_token: str) -> Dict:
    """Fetch a single tweet by ID using X API."""
    url = f"{API_BASE_URL}/{tweet_id}"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tweet {tweet_id}: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

def save_api_response(tweet_id: str, response_data: Dict, output_dir: str):
    """Save API response to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"tweet_{tweet_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)

def append_to_csv(tweet_id: str, text: str, label: int, output_csv: str):
    """Append a row to the output CSV."""
    new_row = pd.DataFrame({
        'tweet_id': [tweet_id],
        'text': [text],
        'label': [label]
    })
    
    if os.path.exists(output_csv):
        new_row.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        new_row.to_csv(output_csv, mode='w', header=True, index=False)

def main():
    # Check if bearer token is set
    if BEARER_TOKEN == "YOUR_BEARER_TOKEN_HERE":
        print("ERROR: Please set your BEARER_TOKEN in the script before running!")
        print("Edit the BEARER_TOKEN variable at the top of fetch_tweets.py")
        return
    
    print("Loading sarcastic tweet IDs...")
    sarcastic_tweet_ids = get_sarcastic_tweet_ids(INPUT_CSV)
    print(f"Found {len(sarcastic_tweet_ids)} sarcastic tweets in dataset")
    
    print("Checking already fetched tweets...")
    already_fetched = load_already_fetched_tweet_ids(OUTPUT_CSV)
    print(f"Already fetched: {len(already_fetched)} tweets")
    
    # Get tweets that haven't been fetched yet
    remaining_tweet_ids = [tid for tid in sarcastic_tweet_ids if tid not in already_fetched]
    print(f"Remaining to fetch: {len(remaining_tweet_ids)} tweets")
    
    if not remaining_tweet_ids:
        print("All sarcastic tweets have already been fetched!")
        return
    
    # Fetch next batch
    batch = remaining_tweet_ids[:BATCH_SIZE]
    print(f"\nFetching {len(batch)} tweets...")
    
    successful = 0
    failed = 0
    
    for i, tweet_id in enumerate(batch, 1):
        print(f"[{i}/{len(batch)}] Fetching tweet {tweet_id}...", end=" ")
        
        response_data = fetch_tweet_by_id(tweet_id, BEARER_TOKEN)
        
        if response_data and 'data' in response_data:
            # Save API response
            save_api_response(tweet_id, response_data, API_RESPONSES_DIR)
            
            # Extract text
            text = response_data['data'].get('text', '')
            
            # Append to CSV
            append_to_csv(tweet_id, text, 1, OUTPUT_CSV)
            
            print("✓ Success")
            successful += 1
        else:
            print("✗ Failed")
            failed += 1
    
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"API responses saved to: {API_RESPONSES_DIR}/")

if __name__ == "__main__":
    main()


import csv
import os

# File paths
truth_file = "datasets/sahil swami dataset/Sarcasm_tweet_truth.txt"
tweets_file = "datasets/sahil swami dataset/Sarcasm_tweets.txt"
output_csv = "datasets/sahil swami dataset/sarcasm_dataset.csv"

# Read truth file and create mapping: tweet_id -> label (1 for YES, 0 for NO)
truth_dict = {}
with open(truth_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            tweet_id = lines[i].strip()
            label = lines[i + 1].strip()
            # Convert YES to 1, NO to 0
            truth_dict[tweet_id] = 1 if label == "YES" else 0

# Read tweets file and create mapping: tweet_id -> text
tweets_dict = {}
with open(tweets_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        tweet_id = lines[i].strip()
        if i + 1 < len(lines):
            text = lines[i + 1].strip()
            if tweet_id and text:  # Only add if both exist
                tweets_dict[tweet_id] = text
            i += 3  # Skip tweet_id, text, and blank line
        else:
            break

# Combine data and create CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['tweet_id', 'label', 'text'])
    
    # Write data for tweets that exist in both files
    for tweet_id in truth_dict:
        if tweet_id in tweets_dict:
            label = truth_dict[tweet_id]
            text = tweets_dict[tweet_id]
            writer.writerow([tweet_id, label, text])

print(f"CSV file created successfully: {output_csv}")
print(f"Total rows: {len([t for t in truth_dict if t in tweets_dict])}")





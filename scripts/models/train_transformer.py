import os
import argparse
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AlbertTokenizer,
)

DATA_PATH = "datasets/combined_dataset.csv"
TEXT_COL = "text"
LABEL_COL = "label"
MAX_LEN = 96
RANDOM_SEED = 42

MODEL_NAMES = {
    "xlm-roberta-base": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "indic-bert": "ai4bharat/indic-bert",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer model for sarcasm detection."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help=(
            "Model to train. You can use a short name "
            f"({', '.join(MODEL_NAMES.keys())}) "
            "or a full Hugging Face model ID (e.g. 'xlm-roberta-base')."
        ),
    )
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


def load_and_split_data():
    df = pd.read_csv(DATA_PATH)

    df = df[[TEXT_COL, LABEL_COL]].dropna()

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[LABEL_COL],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[LABEL_COL],
        random_state=RANDOM_SEED,
    )

    return train_df, val_df, test_df


def make_tokenized_datasets(model_name, train_df, val_df, test_df):
    if "indic-bert" in model_name.lower():
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)

    train_ds = train_ds.rename_column(LABEL_COL, "labels")
    val_ds = val_ds.rename_column(LABEL_COL, "labels")
    test_ds = test_ds.rename_column(LABEL_COL, "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    return tokenizer, train_ds, val_ds, test_ds


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def train_and_eval(model_name, output_dir, train_df, val_df, test_df):
    tokenizer, train_ds, val_ds, test_ds = make_tokenized_datasets(
        model_name, train_df, val_df, test_df
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        save_total_limit=1,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    return test_metrics


def main():
    args = parse_args()
    train_df, val_df, test_df = load_and_split_data()

    os.makedirs("transformer_runs", exist_ok=True)

    print("=" * 70)
    print("TRANSFORMER MODEL TRAINING AND EVALUATION")
    print("="*70)
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print("=" * 70)

    if args.model in MODEL_NAMES:
        short_name = args.model
        hf_name = MODEL_NAMES[args.model]
    else:
        short_name = args.model
        hf_name = args.model

    print(f"\n{'='*70}")
    print(f"MODEL: {short_name.upper()} ({hf_name})")
    print(f"{'='*70}")
    
    try:
        output_dir = os.path.join("transformer_runs", short_name)
        metrics = train_and_eval(hf_name, output_dir, train_df, val_df, test_df)
        
        acc = metrics.get("eval_accuracy", metrics.get("accuracy", 0.0))
        f1 = metrics.get("eval_f1", metrics.get("f1", 0.0))
        prec = metrics.get("eval_precision", metrics.get("precision", 0.0))
        rec = metrics.get("eval_recall", metrics.get("recall", 0.0))
        
        print(f"\n{'─'*70}")
        print(f"TEST SET RESULTS FOR {short_name.upper()}:")
        print(f"{'─'*70}")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
        print(f"{'─'*70}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR training {short_name}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        print(f"{'='*70}\n")
        raise


if __name__ == "__main__":
    main()
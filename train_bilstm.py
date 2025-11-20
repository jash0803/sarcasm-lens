import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_dataset(file_path: str) -> pd.DataFrame:
	"""Load the preprocessed dataset and report basic stats."""
	print(f"Loading dataset from {file_path}...")
	df = pd.read_csv(file_path)
	print(f"Dataset shape: {df.shape}")
	print(f"Columns: {df.columns.tolist()}")
	print(f"Label distribution:\n{df['label'].value_counts()}")
	return df


def prepare_splits(
	texts: np.ndarray,
	labels: np.ndarray,
	test_size: float = 0.2,
	random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Train/test split with stratification."""
	X_train, X_test, y_train, y_test = train_test_split(
		texts,
		labels,
		test_size=test_size,
		random_state=random_state,
		stratify=labels
	)
	print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
	return X_train, X_test, y_train, y_test


def build_bilstm_model(
	vocab_size: int,
	max_len: int,
	embedding_dim: int = 128,
	lstm_units: int = 64,
	dropout_rate: float = 0.5
) -> tf.keras.Model:
	"""Construct a simple Bidirectional LSTM model for binary classification."""
	model = Sequential([
		Embedding(input_dim=vocab_size, output_dim=embedding_dim),
		Bidirectional(LSTM(lstm_units, return_sequences=True)),
		GlobalMaxPooling1D(),
		Dropout(dropout_rate),
		Dense(64, activation='relu'),
		Dropout(dropout_rate),
		Dense(1, activation='sigmoid')
	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
		loss='binary_crossentropy',
		metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
	)
	return model


def compute_max_len_from_texts(tokenizer: Tokenizer, texts: np.ndarray, percentile: float = 95.0) -> int:
	"""Heuristic for sequence length using a percentile of training lengths."""
	sequences = tokenizer.texts_to_sequences(texts)
	lengths = [len(seq) for seq in sequences if len(seq) > 0]
	if not lengths:
		return 20
	max_len = int(np.percentile(lengths, percentile))
	return max(20, min(max_len, 200))  # keep within a reasonable range


def main():
	# Paths
	preprocessed_file = "datasets/sahil swami/sarcasm_dataset_preprocessed.csv"
	model_output_path = "bilstm_model.keras"
	tokenizer_output_path = "tokenizer_bilstm.pkl"
	config_output_path = "bilstm_config.json"

	# Hyperparameters
	vocab_size = 20000  # top-N words to keep
	oov_token = "<OOV>"
	batch_size = 64
	epochs = 10
	validation_split = 0.1

	# Load data
	df = load_dataset(preprocessed_file)
	X = df['text'].astype(str).values
	y = df['label'].astype(int).values

	# Split
	X_train_text, X_test_text, y_train, y_test = prepare_splits(X, y)

	# Tokenize
	print("\nFitting tokenizer...")
	tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
	tokenizer.fit_on_texts(X_train_text)

	# Effective vocab size is min(vocab_size, actual_vocab + 1 for OOV)
	actual_vocab_size = min(vocab_size, len(tokenizer.word_index) + 1)

	# Determine reasonable max_len from train distribution
	max_len = compute_max_len_from_texts(tokenizer, X_train_text, percentile=95.0)
	print(f"Using max_len={max_len}, vocab_size={actual_vocab_size}")

	# Convert to padded sequences
	print("Tokenizing and padding sequences...")
	X_train_seq = tokenizer.texts_to_sequences(X_train_text)
	X_test_seq = tokenizer.texts_to_sequences(X_test_text)
	X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
	X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

	# Build model
	print("\nBuilding Bidirectional LSTM model...")
	model = build_bilstm_model(actual_vocab_size, max_len)
	model.summary(print_fn=lambda x: print(x))

	# Callbacks
	early_stop = EarlyStopping(
		monitor='val_loss',
		patience=2,
		restore_best_weights=True
	)
	checkpoint = ModelCheckpoint(
		model_output_path,
		monitor='val_loss',
		save_best_only=True,
		save_weights_only=False
	)

	# Train
	print("\nTraining...")
	history = model.fit(
		X_train_pad,
		y_train,
		validation_split=validation_split,
		epochs=epochs,
		batch_size=batch_size,
		callbacks=[early_stop, checkpoint],
		verbose=1
	)

	# Evaluate
	print("\nEvaluating on test set...")
	test_probs = model.predict(X_test_pad, batch_size=batch_size, verbose=0).ravel()
	y_pred = (test_probs >= 0.5).astype(int)

	acc = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred, average='weighted')
	print(f"\nTest Accuracy: {acc:.4f}")
	print(f"Test F1 (weighted): {f1:.4f}")

	print("\n" + "="*50)
	print("Detailed Classification Report:")
	print("="*50)
	print(classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic']))

	print("\n" + "="*50)
	print("Confusion Matrix:")
	print("="*50)
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	print(f"\nTrue Negatives: {cm[0][0]}")
	print(f"False Positives: {cm[0][1]}")
	print(f"False Negatives: {cm[1][0]}")
	print(f"True Positives: {cm[1][1]}")

	# Save artifacts
	print("\nSaving model and tokenizer...")
	# Model already saved via checkpoint; ensure existence or save final
	if not os.path.exists(model_output_path):
		model.save(model_output_path)
	joblib.dump(tokenizer, tokenizer_output_path)

	with open(config_output_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"max_len": int(max_len),
				"vocab_size": int(actual_vocab_size),
				"oov_token": oov_token
			},
			f,
			indent=2
		)

	print(f"Model saved to '{model_output_path}'")
	print(f"Tokenizer saved to '{tokenizer_output_path}'")
	print(f"Config saved to '{config_output_path}'")
	print("\nTraining complete!")


if __name__ == "__main__":
	main()



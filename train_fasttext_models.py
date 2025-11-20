import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import fasttext
import os
import tempfile

def load_and_prepare_data(file_path):
    """Load the preprocessed dataset."""
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

def prepare_fasttext_data(texts, output_file):
    """Prepare text data in FastText format (one sentence per line)."""
    print(f"\nPreparing FastText training data...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            # FastText expects one sentence per line, preprocessed
            cleaned_text = ' '.join(str(text).split())  # Normalize whitespace
            f.write(cleaned_text + '\n')
    print(f"FastText training data saved to {output_file}")

def train_fasttext_model(training_file, model_dim=100, min_count=1, epoch=10):
    """Train a FastText unsupervised model."""
    print(f"\nTraining FastText model...")
    print(f"Parameters: dim={model_dim}, min_count={min_count}, epoch={epoch}")
    
    # Train FastText model (unsupervised, skipgram)
    model = fasttext.train_unsupervised(
        training_file,
        model='skipgram',  # or 'cbow'
        dim=model_dim,
        minCount=min_count,
        epoch=epoch,
        lr=0.05,
        wordNgrams=1
    )
    
    print(f"FastText model trained successfully!")
    print(f"Vocabulary size: {len(model.get_words())}")
    
    return model

def get_sentence_embedding(model, text, dim=100):
    """Get sentence embedding by averaging word embeddings."""
    words = str(text).split()
    word_vectors = []
    
    for word in words:
        if word in model.get_words():
            word_vectors.append(model.get_word_vector(word))
    
    if len(word_vectors) == 0:
        # Return zero vector if no words found
        return np.zeros(dim)
    
    # Average word vectors to get sentence embedding
    return np.mean(word_vectors, axis=0)

def extract_embeddings(model, texts, dim=100):
    """Extract FastText embeddings for all texts."""
    print(f"\nExtracting FastText embeddings...")
    embeddings = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(texts)} texts...")
        embedding = get_sentence_embedding(model, text, dim)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=42):
    """Train a Random Forest classifier."""
    print("\nTraining Random Forest classifier with FastText embeddings...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Initialize Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train the model
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = rf_classifier.predict(X_train)
    y_test_pred = rf_classifier.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training F1 Score: {train_f1:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    return rf_classifier, y_test_pred

def train_svm(X_train, y_train, X_test, y_test, C=1.0, random_state=42, max_iter=1000):
    """Train a Linear Support Vector Machine classifier."""
    print("\nTraining Linear SVM classifier with FastText embeddings...")
    print(f"Parameters: C={C}, max_iter={max_iter}")
    
    # Initialize Linear SVM
    svm_classifier = LinearSVC(
        C=C,
        random_state=random_state,
        max_iter=max_iter,
        class_weight='balanced',  # Handle class imbalance
        dual=False  # Use primal formulation for large datasets
    )
    
    # Train the model
    svm_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = svm_classifier.predict(X_train)
    y_test_pred = svm_classifier.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training F1 Score: {train_f1:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    return svm_classifier, y_test_pred

def evaluate_model(y_test, y_pred, model_name):
    """Print detailed evaluation metrics."""
    print("\n" + "="*50)
    print(f"{model_name} - Detailed Classification Report:")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic']))
    
    print("\n" + "="*50)
    print(f"{model_name} - Confusion Matrix:")
    print("="*50)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

def main():
    # File paths
    preprocessed_file = "datasets/combined_dataset.csv"
    
    # FastText parameters
    fasttext_dim = 100
    fasttext_min_count = 1
    fasttext_epoch = 10
    
    # Load data
    df = load_and_prepare_data(preprocessed_file)
    
    # Prepare features and labels
    X = df['text'].values
    y = df['label'].values
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Prepare FastText training data (use all data for better embeddings)
    print("\n" + "="*50)
    print("Step 1: Training FastText Model")
    print("="*50)
    
    # Create temporary file for FastText training
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
        fasttext_training_file = tmp_file.name
        prepare_fasttext_data(X, fasttext_training_file)
    
    try:
        # Train FastText model
        fasttext_model = train_fasttext_model(
            fasttext_training_file,
            model_dim=fasttext_dim,
            min_count=fasttext_min_count,
            epoch=fasttext_epoch
        )
        
        # Extract embeddings for train and test sets
        print("\n" + "="*50)
        print("Step 2: Extracting FastText Embeddings")
        print("="*50)
        
        X_train_embeddings = extract_embeddings(fasttext_model, X_train, dim=fasttext_dim)
        X_test_embeddings = extract_embeddings(fasttext_model, X_test, dim=fasttext_dim)
        
        print(f"\nTrain embeddings shape: {X_train_embeddings.shape}")
        print(f"Test embeddings shape: {X_test_embeddings.shape}")
        
        # Train Random Forest with FastText embeddings
        print("\n" + "="*50)
        print("Step 3: Training Random Forest with FastText Embeddings")
        print("="*50)
        
        rf_model, rf_y_pred = train_random_forest(
            X_train_embeddings, y_train,
            X_test_embeddings, y_test,
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        
        evaluate_model(y_test, rf_y_pred, "Random Forest")
        
        # Train SVM with FastText embeddings
        print("\n" + "="*50)
        print("Step 4: Training SVM with FastText Embeddings")
        print("="*50)
        
        svm_model, svm_y_pred = train_svm(
            X_train_embeddings, y_train,
            X_test_embeddings, y_test,
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        
        evaluate_model(y_test, svm_y_pred, "SVM")
        
        # Save models
        print("\n" + "="*50)
        print("Saving models...")
        print("="*50)
        
        # Save FastText model
        fasttext_model_path = 'fasttext_model.bin'
        fasttext_model.save_model(fasttext_model_path)
        print(f"FastText model saved as '{fasttext_model_path}'")
        
        # Save Random Forest model
        joblib.dump(rf_model, 'random_forest_fasttext_model.pkl')
        print("Random Forest model saved as 'random_forest_fasttext_model.pkl'")
        
        # Save SVM model
        joblib.dump(svm_model, 'svm_fasttext_model.pkl')
        print("SVM model saved as 'svm_fasttext_model.pkl'")
        
        print("\nTraining complete!")
        
    finally:
        # Clean up temporary file
        if os.path.exists(fasttext_training_file):
            os.remove(fasttext_training_file)
            print(f"\nCleaned up temporary file: {fasttext_training_file}")

if __name__ == "__main__":
    main()


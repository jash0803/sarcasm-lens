import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import joblib
import re

def load_and_prepare_data(file_path):
    """Load the preprocessed dataset."""
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=42):
    """Train a Random Forest classifier."""
    print("\nTraining Random Forest classifier...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_classifier.fit(X_train, y_train)
    
    y_train_pred = rf_classifier.predict(X_train)
    y_test_pred = rf_classifier.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training F1 Score: {train_f1:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    return rf_classifier, y_test_pred

def main():
    preprocessed_file = "datasets/combined_dataset.csv"
    
    df = load_and_prepare_data(preprocessed_file)
    
    X = df['text'].values
    y = df['label'].values
    
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    print("\nVectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=None
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")
    print(f"TF-IDF matrix shape (test): {X_test_tfidf.shape}")
    
    rf_model, y_pred = train_random_forest(
        X_train_tfidf, y_train, 
        X_test_tfidf, y_test,
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    
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
    
    print("\n" + "="*50)
    print("Top 20 Most Important Features:")
    print("="*50)
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    print("\n" + "="*50)
    print("Saving model and vectorizer...")
    print("="*50)
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Model saved as 'random_forest_model.pkl'")
    print("Vectorizer saved as 'tfidf_vectorizer.pkl'")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Training Multinomial Naive Bayes dengan TF-IDF.
"""

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Memuat data training dan testing."""
    print("Memuat dataset...")
    
    train_df = pd.read_csv("processed/train.csv")
    test_df = pd.read_csv("processed/test.csv")
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Testing data: {len(test_df)} samples")
    
    return train_df, test_df

def create_pipeline():
    """
    Membuat pipeline TF-IDF + Multinomial Naive Bayes.
    """
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=5000,           # 5000 fitur terbaik
        ngram_range=(1, 2),          # unigrams dan bigrams
        stop_words='english',        # hapus stop words
        min_df=2,                    # minimal muncul 2 kali
        max_df=0.95                  # maksimal di 95% dokumen
    )
    
    # Multinomial Naive Bayes
    nb = MultinomialNB(alpha=0.1)    # smoothing parameter
    
    # Pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('nb', nb)
    ])
    
    return pipeline

def train_model(pipeline, X_train, y_train):
    """Melatih model."""
    print("\nTraining Multinomial Naive Bayes dengan TF-IDF...")
    
    # Label mapping
    label_mapping = {'Information': 0, 'Request': 1, 'Problem': 2}
    y_train_encoded = y_train.map(label_mapping)
    
    # Train
    pipeline.fit(X_train, y_train_encoded)
    
    print("Training selesai!")
    
    return pipeline, label_mapping

def evaluate_model(pipeline, X_test, y_test, label_mapping):
    """Evaluasi model."""
    print("\n=== Evaluasi Model ===")
    
    # Encode labels
    y_test_encoded = y_test.map(label_mapping)
    
    # Predict
    y_pred_encoded = pipeline.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    y_pred = [inverse_mapping[pred] for pred in y_pred_encoded]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Information', 'Request', 'Problem']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    print("Confusion Matrix:")
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Information', 'Request', 'Problem'],
                yticklabels=['Information', 'Request', 'Problem'])
    plt.title('Confusion Matrix - Multinomial Naive Bayes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png')
    print("\nConfusion matrix saved to models/confusion_matrix.png")
    
    return accuracy, cm

def save_model(pipeline, label_mapping):
    """Menyimpan model."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "customer_classifier.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': pipeline,
            'label_mapping': label_mapping,
            'tfidf_features': pipeline.named_steps['tfidf'].get_feature_names_out().tolist()
        }, f)
    
    print(f"\nModel saved to {model_path}")
    
    # Save label mapping
    mapping_path = os.path.join(models_dir, "label_mapping.txt")
    with open(mapping_path, 'w') as f:
        for label, code in label_mapping.items():
            f.write(f"{label}: {code}\n")
    
    print(f"Label mapping saved to {mapping_path}")

def test_examples(pipeline, label_mapping):
    """Test dengan contoh-contoh."""
    print("\n=== Contoh Prediksi ===")
    
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    
    examples = [
        "Hi Support, Where is your headquarters located?",
        "Hi Support, I need to reset my password",
        "Hi Support, The application crashes every time I open the settings tab",
        "Hi Support, How do I upgrade to the Enterprise plan?",
        "Hi Support, Please send me the invoice for last month",
        "Hi Support, My internet connection is very slow",
        "Hi Support, I want to cancel my subscription"
    ]
    
    for i, example in enumerate(examples):
        prediction_encoded = pipeline.predict([example])[0]
        prediction = inverse_mapping[prediction_encoded]
        
        print(f"\nContoh {i+1}:")
        print(f"  Text: {example}")
        print(f"  Prediction: {prediction}")
        
        # Get probabilities
        try:
            probabilities = pipeline.predict_proba([example])[0]
            print(f"  Probabilities:")
            for j, label in enumerate(['Information', 'Request', 'Problem']):
                print(f"    {label}: {probabilities[j]:.4f}")
        except:
            pass

def main():
    print("=== Training Multinomial Naive Bayes dengan TF-IDF ===")
    
    # Load data
    train_df, test_df = load_data()
    
    # Prepare data
    X_train = train_df['Cleaned_Text']
    y_train = train_df['Label']
    X_test = test_df['Cleaned_Text']
    y_test = test_df['Label']
    
    # Create and train pipeline
    pipeline = create_pipeline()
    pipeline, label_mapping = train_model(pipeline, X_train, y_train)
    
    # Evaluate
    accuracy, cm = evaluate_model(pipeline, X_test, y_test, label_mapping)
    
    # Save model
    save_model(pipeline, label_mapping)
    
    # Test examples
    test_examples(pipeline, label_mapping)
    
    # Summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: Multinomial Naive Bayes")
    print(f"Feature: TF-IDF (5000 features, unigrams+bigrams)")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    if accuracy == 1.0:
        print("\n✅ Model mencapai akurasi sempurna!")
    elif accuracy >= 0.95:
        print("\n✅ Model sangat akurat!")
    elif accuracy >= 0.90:
        print("\n✅ Model akurat!")
    else:
        print("\n⚠️  Model perlu improvement.")

if __name__ == "__main__":
    main()
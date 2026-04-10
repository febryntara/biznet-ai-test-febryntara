#!/usr/bin/env python3

import pandas as pd
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load data dari file CSV."""    
    train_df = pd.read_csv("processed/train.csv")
    test_df = pd.read_csv("processed/test.csv")
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Testing data: {len(test_df)} samples")
    
    # Distribusi label
    print("\nDistribusi Label Training:")
    train_dist = train_df['Label'].value_counts()
    for label, count in train_dist.items():
        percentage = (count / len(train_df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print("\nDistribusi Label Testing:")
    test_dist = test_df['Label'].value_counts()
    for label, count in test_dist.items():
        percentage = (count / len(test_df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    return train_df, test_df

def create_pipeline():
    """
    Membuat pipeline TF-IDF + Multinomial Naive Bayes.
    """
    # TF-IDF Vectorizer dengan optimasi untuk text ISP tickets
    tfidf = TfidfVectorizer(
        max_features=3500,           # 3500 fitur terbaik
        ngram_range=(1, 2),          # unigrams, bigrams, dan trigrams
        stop_words='english',        # hapus stop words
        min_df=1,                    # minimal muncul 1 kali
        max_df=0.9,                  # maksimal di 90% dokumen
        sublinear_tf=True,           # gunakan sublinear TF scaling
        analyzer='word',             # analisis per kata
        token_pattern=r'(?u)\b\w+\b' # token pattern
    )
    
    # Multinomial Naive Bayes dengan parameter tuning
    nb = MultinomialNB(
        alpha=0.5,      # smoothing parameter
        fit_prior=True  # learn class prior probabilities
    )
    
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
    
    # Feature importance analysis
    try:
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
        nb_coef = pipeline.named_steps['nb'].coef_
        
        print("\nTop 10 features per class:")
        for i, class_name in enumerate(['Information', 'Request', 'Problem']):
            top_indices = np.argsort(nb_coef[i])[-10:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            print(f"\n{class_name}:")
            print("  " + ", ".join(top_features))
    except:
        pass
    
    return pipeline, label_mapping

def evaluate_model(pipeline, X_test, y_test, label_mapping):
    """Evaluasi model secara komprehensif."""
    print("\n=== Evaluasi Model ===")
    
    # Encode labels
    y_test_encoded = y_test.map(label_mapping)
    
    # Predict
    y_pred_encoded = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
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
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class metrics
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(['Information', 'Request', 'Problem']):
        class_mask = y_test_encoded == i
        if sum(class_mask) > 0:
            class_accuracy = accuracy_score(
                y_test_encoded[class_mask], 
                y_pred_encoded[class_mask]
            )
            print(f"  {class_name}: {class_accuracy:.4f} ({sum(class_mask)} samples)")
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Information', 'Request', 'Problem'],
                yticklabels=['Information', 'Request', 'Problem'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png')
    print("\nConfusion matrix saved to models/confusion_matrix.png")
    
    # Probability distribution analysis
    print("\n=== Probability Distribution ===")
    for i, class_name in enumerate(['Information', 'Request', 'Problem']):
        class_probs = y_pred_proba[y_test_encoded == i, i]
        if len(class_probs) > 0:
            print(f"{class_name}: Mean confidence = {class_probs.mean():.4f}, Std = {class_probs.std():.4f}")
    
    return accuracy, cm

def save_model(pipeline, label_mapping, feature_names=None):
    """Menyimpan model."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "customer_classifier.pkl")
    model_data = {
        'model': pipeline,
        'label_mapping': label_mapping,
        'feature_names': feature_names if feature_names else []
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {model_path}")
    
    # Save label mapping
    mapping_path = os.path.join(models_dir, "label_mapping.txt")
    with open(mapping_path, 'w') as f:
        for label, code in label_mapping.items():
            f.write(f"{label}: {code}\n")
    
    print(f"Label mapping saved to {mapping_path}")

def test_examples(pipeline, label_mapping):
    """Test dengan contoh-contoh ISP tickets (expected vs predicted)."""
    print("\n=== Contoh Prediksi ISP Tickets ===")

    inverse_mapping = {v: k for k, v in label_mapping.items()}

    # Format: (teks, label_yang_benar)
    examples = [
        # Information
        ("Where is your headquarters located?",      "Information"),
        ("What are your business hours?",            "Information"),
        ("How do I check my data usage?",            "Information"),
        ("Does your service cover South Jakarta?",   "Information"),
        ("What is the price for Business 100Mbps?",  "Information"),
        # Request
        ("I need to reset my password",              "Request"),
        ("Please send me last month's invoice",      "Request"),
        ("Can you install new connection at my address?", "Request"),
        ("I want to cancel my subscription",         "Request"),
        ("Need technician to check fiber optic line","Request"),
        # Problem
        ("The application crashes every time",       "Problem"),
        ("My internet is very slow today",           "Problem"),
        ("I was charged twice for this month",       "Problem"),
        ("Cannot login to my account",               "Problem"),
        ("Modem keeps disconnecting every few minutes", "Problem"),
    ]

    correct = 0
    print(f"\n{'Status':<6} {'Expected':<13} {'Predicted':<13} {'Conf':>6}  Text")
    print("-" * 75)

    for text, expected in examples:
        pred_encoded = pipeline.predict([text])[0]
        predicted    = inverse_mapping[pred_encoded]
        confidence   = pipeline.predict_proba([text])[0][pred_encoded]
        status       = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        short_text = text[:45] + "..." if len(text) > 45 else text
        print(f"  {status}    {expected:<13} {predicted:<13} {confidence:>5.2f}  {short_text}")

    print("-" * 75)
    print(f"  Contoh benar: {correct}/{len(examples)}")

def cross_validate_model(X, y, label_mapping):
    """Cross-validation untuk evaluasi robust."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    print("\n=== Cross-Validation ===")
    
    # Create pipeline for CV
    pipeline_cv = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3500, ngram_range=(1, 2), stop_words='english', min_df=1)),
        ('nb', MultinomialNB(alpha=0.5))
    ])
    
    # Encode labels
    y_encoded = y.map(label_mapping)
    
    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_cv, X, y_encoded, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores.mean()

def main():
    print("=== Training Multinomial Naive Bayes ===")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Prepare data
    X_train = train_df['Cleaned_Text']
    y_train = train_df['Label']
    X_test = test_df['Cleaned_Text']
    y_test = test_df['Label']
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create and train pipeline
    pipeline = create_pipeline()
    pipeline, label_mapping = train_model(pipeline, X_train, y_train)
    
    # Get feature names
    try:
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out().tolist()
    except:
        feature_names = None
    
    # Evaluate
    accuracy, cm = evaluate_model(pipeline, X_test, y_test, label_mapping)
    
    # Cross-validation
    cv_score = cross_validate_model(pd.concat([X_train, X_test]), 
                                   pd.concat([y_train, y_test]), 
                                   label_mapping)
    
    # Save model
    save_model(pipeline, label_mapping, feature_names)
    
    # Test examples
    test_examples(pipeline, label_mapping)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: Multinomial Naive Bayes")
    print(f"Feature: TF-IDF (3500 features, ngram_range=(1,2))")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Mean Accuracy: {cv_score:.4f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    if accuracy >= 0.95:
        print("\n✅ Model sangat akurat dengan dataset!")
    elif accuracy >= 0.90:
        print("\n✅ Model akurat dengan dataset!")
    else:
        print("\n⚠️  Model perlu improvement.")
    
    print("\nUntuk prediksi:")
    print("  python predict.py --text \"Your message here\"")

if __name__ == "__main__":
    main()
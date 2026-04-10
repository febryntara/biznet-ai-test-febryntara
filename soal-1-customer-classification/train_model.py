#!/usr/bin/env python3

import pandas as pd
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
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

def run_grid_search(X_train, y_train_encoded):
    """
    Jalankan GridSearchCV untuk cari parameter terbaik.
    """
    print("\n=== GridSearch: Mencari Parameter Terbaik ===")
    print("Ini mungkin butuh beberapa menit...\n")

    base_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )),
        ('lr', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ))
    ])

    param_grid = {
        'tfidf__max_features': [3500, 5000],
        'tfidf__ngram_range' : [(1, 2), (1, 3)],
        'tfidf__min_df'      : [1, 2],
        'tfidf__max_df'      : [0.90, 0.85],
        'lr__C'              : [0.5, 1.0, 5.0, 10.0],
        'lr__class_weight'   : [None, 'balanced'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        refit=True   # otomatis refit pakai best params
    )

    grid_search.fit(X_train, y_train_encoded)

    print("\n✅ GridSearch selesai!")
    print(f"   Best CV Accuracy : {grid_search.best_score_:.4f}")
    print(f"   Best Parameters  :")
    for k, v in grid_search.best_params_.items():
        print(f"     {k}: {v}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def create_pipeline(best_params=None):
    """
    Membuat pipeline TF-IDF + Logistic Regression.
    Jika best_params tersedia (dari GridSearch), pakai itu.
    Kalau tidak, pakai default.
    """
    if best_params:
        tfidf = TfidfVectorizer(
            max_features  = best_params.get('tfidf__max_features', 5000),
            ngram_range   = best_params.get('tfidf__ngram_range',   (1, 2)),
            stop_words    = 'english',
            min_df        = best_params.get('tfidf__min_df',        1),
            max_df        = best_params.get('tfidf__max_df',        0.90),
            sublinear_tf  = True,
            analyzer      = 'word',
            token_pattern = r'(?u)\b\w+\b'
        )
        lr = LogisticRegression(
            C            = best_params.get('lr__C',            1.0),
            class_weight = best_params.get('lr__class_weight', None),
            max_iter     = 1000,
            solver       = 'lbfgs',
            random_state = 42
        )
    else:
        # Default fallback
        tfidf = TfidfVectorizer(
            max_features  = 5000,
            ngram_range   = (1, 2),
            stop_words    = 'english',
            min_df        = 1,
            max_df        = 0.90,
            sublinear_tf  = True,
            analyzer      = 'word',
            token_pattern = r'(?u)\b\w+\b'
        )
        lr = LogisticRegression(
            C            = 1.0,
            max_iter     = 1000,
            solver       = 'lbfgs',
            random_state = 42
        )

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('lr',    lr)
    ])

    return pipeline

def train_model(pipeline, X_train, y_train):
    """Melatih model dengan pipeline yang sudah dikonfigurasi."""
    print("\nTraining Logistic Regression dengan TF-IDF...")
    
    # Label mapping
    label_mapping = {'Information': 0, 'Request': 1, 'Problem': 2}
    y_train_encoded = y_train.map(label_mapping)
    
    # Train
    pipeline.fit(X_train, y_train_encoded)
    
    print("Training selesai!")
    
    # Feature importance analysis
    try:
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
        lr_coef = pipeline.named_steps['lr'].coef_

        print("\nTop 10 features per class:")
        for i, class_name in enumerate(['Information', 'Request', 'Problem']):
            top_indices = np.argsort(lr_coef[i])[-10:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            print(f"\n{class_name}:")
            print("  " + ", ".join(top_features))
    except Exception:
        pass
    
    return pipeline, label_mapping

def evaluate_model(pipeline, X_test, y_test, label_mapping):
    """Evaluasi model secara komprehensif."""
    print("\n=== Evaluasi Model ===")
    
    # Encode labels
    y_test_encoded = y_test.map(label_mapping)
    
    # Predict
    y_pred_encoded = pipeline.predict(X_test)
    y_pred_proba   = pipeline.predict_proba(X_test)
    
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
    plt.title('Confusion Matrix (GridSearch Optimized)')
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

def save_model(pipeline, label_mapping, best_params=None, feature_names=None):
    """Menyimpan model."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "customer_classifier.pkl")
    model_data = {
        'model'         : pipeline,
        'label_mapping' : label_mapping,
        'feature_names' : feature_names if feature_names else [],
        'best_params'   : best_params if best_params else {}
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

    # Save best params (jika ada)
    if best_params:
        params_path = os.path.join(models_dir, "best_params.txt")
        with open(params_path, 'w') as f:
            f.write("Best Parameters dari GridSearchCV:\n")
            f.write("=" * 40 + "\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"Best params saved to {params_path}")

def test_examples(pipeline, label_mapping):
    """Test dengan contoh-contoh ISP tickets (expected vs predicted)."""
    print("\n=== Contoh Prediksi ISP Tickets ===")

    inverse_mapping = {v: k for k, v in label_mapping.items()}

    # Format: (teks, label_yang_benar)
    examples = [
        # Information
        ("Where is your headquarters located?",           "Information"),
        ("What are your business hours?",                 "Information"),
        ("How do I check my data usage?",                 "Information"),
        ("Does your service cover South Jakarta?",        "Information"),
        ("What is the price for Business 100Mbps?",       "Information"),
        # Request
        ("I need to reset my password",                   "Request"),
        ("Please send me last month's invoice",           "Request"),
        ("Can you install new connection at my address?", "Request"),
        ("I want to cancel my subscription",              "Request"),
        ("Need technician to check fiber optic line",     "Request"),
        # Problem
        ("The application crashes every time",            "Problem"),
        ("My internet is very slow today",                "Problem"),
        ("I was charged twice for this month",            "Problem"),
        ("Cannot login to my account",                    "Problem"),
        ("Modem keeps disconnecting every few minutes",   "Problem"),
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

def cross_validate_model(X, y, label_mapping, best_params=None):
    """Cross-validation untuk evaluasi robust menggunakan best params."""
    print("\n=== Cross-Validation (Best Params) ===")

    pipeline_cv = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features  = best_params.get('tfidf__max_features', 5000) if best_params else 5000,
            ngram_range   = best_params.get('tfidf__ngram_range',  (1, 2)) if best_params else (1, 2),
            stop_words    = 'english',
            min_df        = best_params.get('tfidf__min_df',       1) if best_params else 1,
            max_df        = best_params.get('tfidf__max_df',       0.90) if best_params else 0.90,
            sublinear_tf  = True,
        )),
        ('lr', LogisticRegression(
            C            = best_params.get('lr__C',            1.0) if best_params else 1.0,
            class_weight = best_params.get('lr__class_weight', None) if best_params else None,
            max_iter     = 1000,
            solver       = 'lbfgs',
            random_state = 42
        ))
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
    print("=== Training Logistic Regression + GridSearch ===")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Prepare data
    X_train = train_df['Cleaned_Text']
    y_train = train_df['Label']
    X_test  = test_df['Cleaned_Text']
    y_test  = test_df['Label']
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Label mapping (dibutuhkan sebelum GridSearch)
    label_mapping   = {'Information': 0, 'Request': 1, 'Problem': 2}
    y_train_encoded = y_train.map(label_mapping)

    # ── GRID SEARCH ────────────────────────────────────────────────────────────
    best_pipeline, best_params, best_cv_score = run_grid_search(X_train, y_train_encoded)
    # ── END GRID SEARCH ────────────────────────────────────────────────────────

    # Gunakan best_pipeline langsung (sudah di-refit oleh GridSearchCV)
    pipeline = best_pipeline

    # Feature importance analysis
    print("\nTop 10 features per class (best model):")
    try:
        feature_names_arr = pipeline.named_steps['tfidf'].get_feature_names_out()
        lr_coef           = pipeline.named_steps['lr'].coef_
        for i, class_name in enumerate(['Information', 'Request', 'Problem']):
            top_indices  = np.argsort(lr_coef[i])[-10:][::-1]
            top_features = [feature_names_arr[idx] for idx in top_indices]
            print(f"\n{class_name}:")
            print("  " + ", ".join(top_features))
        feature_names = feature_names_arr.tolist()
    except Exception:
        feature_names = None

    # Evaluate
    accuracy, cm = evaluate_model(pipeline, X_test, y_test, label_mapping)
    
    # Cross-validation dengan best params
    cv_score = cross_validate_model(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        label_mapping,
        best_params=best_params
    )
    
    # Save model
    save_model(pipeline, label_mapping, best_params=best_params, feature_names=feature_names)
    
    # Test examples
    test_examples(pipeline, label_mapping)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model            : Logistic Regression + GridSearchCV")
    print(f"Best CV Score    : {best_cv_score:.4f}")
    print(f"Test Accuracy    : {accuracy:.4f}")
    print(f"CV Mean Accuracy : {cv_score:.4f}")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"\nBest Parameters  :")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
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

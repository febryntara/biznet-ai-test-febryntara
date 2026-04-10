#!/usr/bin/env python3
"""
Script prediksi dengan adaptive learning.
Model akan bertanya feedback dan melakukan retraining jika diperlukan.
"""

import pickle
import re
import sys
import pandas as pd
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def clean_text(text):
    """
    Membersihkan teks dari noise sintetis.
    """
    if not isinstance(text, str):
        return ""
    
    # Hapus "Hi Support, " dari awal (case insensitive)
    text = re.sub(r'^hi support,\s*', '', text, flags=re.IGNORECASE)
    
    # Cari posisi tanda titik "." atau tanda tanya "?" pertama
    dot_pos = text.find('.')
    question_pos = text.find('?')
    
    # Tentukan posisi akhir kalimat asli
    end_positions = [pos for pos in [dot_pos, question_pos] if pos != -1]
    
    if end_positions:
        # Ambil bagian sebelum tanda baca pertama
        end_pos = min(end_positions)
        cleaned = text[:end_pos + 1].strip()
    else:
        # Jika tidak ada tanda baca, ambil seluruh teks
        cleaned = text.strip()
    
    return cleaned

def load_model():
    """Memuat model yang sudah disimpan."""
    try:
        with open('models/customer_classifier.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        label_mapping = model_data['label_mapping']
        
        # Create inverse mapping
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        
        return model, label_mapping, inverse_mapping
    except FileNotFoundError:
        print("Error: Model file not found.")
        print("Please train the model first:")
        print("  python preprocess.py")
        print("  python train_model.py")
        sys.exit(1)

def load_feedback_data():
    """Memuat data feedback yang sudah terkumpul."""
    feedback_path = "models/feedback_data.csv"
    if os.path.exists(feedback_path):
        return pd.read_csv(feedback_path)
    else:
        # Create empty dataframe
        return pd.DataFrame(columns=['text', 'cleaned_text', 'prediction', 'correct_label', 'timestamp'])

def save_feedback(text, cleaned_text, prediction, correct_label):
    """Menyimpan feedback ke file."""
    feedback_path = "models/feedback_data.csv"
    
    # Load existing data
    if os.path.exists(feedback_path):
        df = pd.read_csv(feedback_path)
    else:
        df = pd.DataFrame(columns=['text', 'cleaned_text', 'prediction', 'correct_label', 'timestamp'])
    
    # Add new feedback
    new_row = {
        'text': text,
        'cleaned_text': cleaned_text,
        'prediction': prediction,
        'correct_label': correct_label,
        'timestamp': datetime.now().isoformat()
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(feedback_path, index=False)
    
    print(f"Feedback saved. Total feedback samples: {len(df)}")
    return df

def retrain_model():
    """Retrain model dengan data feedback."""
    print("\n=== Retraining Model dengan Feedback ===")
    
    # Load original training data
    train_df = pd.read_csv("processed/train.csv")
    
    # Load feedback data
    feedback_df = load_feedback_data()
    
    if len(feedback_df) == 0:
        print("No feedback data available for retraining.")
        return False
    
    print(f"Original training data: {len(train_df)} samples")
    print(f"Feedback data: {len(feedback_df)} samples")
    
    # Combine data
    combined_df = pd.concat([
        train_df[['Cleaned_Text', 'Label']].rename(columns={'Cleaned_Text': 'text', 'Label': 'label'}),
        feedback_df[['cleaned_text', 'correct_label']].rename(columns={'cleaned_text': 'text', 'correct_label': 'label'})
    ], ignore_index=True)
    
    print(f"Combined training data: {len(combined_df)} samples")
    
    # Prepare data
    X = combined_df['text']
    y = combined_df['label']
    
    # Label mapping
    label_mapping = {'Information': 0, 'Request': 1, 'Problem': 2}
    y_encoded = y.map(label_mapping)
    
    # Create new pipeline
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=2,
        max_df=0.85
    )
    
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
    pipeline = Pipeline([('tfidf', tfidf), ('lr', lr)])
    
    # Retrain
    print("Retraining model...")
    pipeline.fit(X, y_encoded)
    
    # Save retrained model
    model_path = "models/customer_classifier_retrained.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': pipeline,
            'label_mapping': label_mapping,
            'training_date': datetime.now().isoformat(),
            'original_samples': len(train_df),
            'feedback_samples': len(feedback_df)
        }, f)
    
    print(f"Retrained model saved to {model_path}")
    
    # Also update the main model
    main_model_path = "models/customer_classifier.pkl"
    with open(main_model_path, 'wb') as f:
        pickle.dump({
            'model': pipeline,
            'label_mapping': label_mapping
        }, f)
    
    print(f"Main model updated at {main_model_path}")
    
    return True

def predict_with_feedback():
    """Mode adaptive learning dengan feedback."""
    print("=== Adaptive Learning Mode ===")
    print("Model akan belajar dari feedback Anda.")
    print("Setelah prediksi, Anda akan diminta:")
    print("  y = Prediction benar")
    print("  n = Prediction salah (akan diminta label yang benar)")
    print("  q = Keluar")
    print("-" * 50)
    
    # Load model
    model, label_mapping, inverse_mapping = load_model()
    
    feedback_count = 0
    
    while True:
        print("\n" + "="*50)
        text = input("Masukkan pesan pelanggan (atau 'q' untuk keluar): ").strip()
        
        if text.lower() == 'q':
            break
        
        if not text:
            continue
        
        # Clean and predict
        cleaned_text = clean_text(text)
        prediction_encoded = model.predict([cleaned_text])[0]
        prediction = inverse_mapping[prediction_encoded]
        
        # Get probabilities
        try:
            probabilities = model.predict_proba([cleaned_text])[0]
            prob_dict = {}
            for code, label in inverse_mapping.items():
                prob_dict[label] = probabilities[code]
        except:
            prob_dict = None
        
        print(f"\nCleaned Text: {cleaned_text}")
        print(f"Prediction: {prediction}")
        
        if prob_dict:
            print("Probabilities:")
            for label, prob in prob_dict.items():
                print(f"  {label}: {prob:.4f}")
        
        # Ask for feedback
        while True:
            feedback = input("\nApakah prediction benar? (y/n/q): ").strip().lower()
            
            if feedback == 'q':
                print("Keluar dari adaptive learning mode.")
                return
            
            if feedback == 'y':
                # Prediction correct, save as positive feedback
                save_feedback(text, cleaned_text, prediction, prediction)
                print("✅ Terima kasih! Feedback disimpan.")
                feedback_count += 1
                break
            
            elif feedback == 'n':
                # Prediction wrong, ask for correct label
                print("\nKategori yang benar:")
                print("  1. Information")
                print("  2. Request")
                print("  3. Problem")
                
                while True:
                    try:
                        correct_choice = input("Pilih kategori (1/2/3): ").strip()
                        if correct_choice == '1':
                            correct_label = 'Information'
                            break
                        elif correct_choice == '2':
                            correct_label = 'Request'
                            break
                        elif correct_choice == '3':
                            correct_label = 'Problem'
                            break
                        else:
                            print("Pilihan tidak valid. Masukkan 1, 2, atau 3.")
                    except:
                        print("Input tidak valid.")
                
                # Save feedback with correct label
                save_feedback(text, cleaned_text, prediction, correct_label)
                print(f"✅ Feedback disimpan. Label benar: {correct_label}")
                feedback_count += 1
                
                # Ask if want to retrain now
                retrain_now = input("\nRetrain model sekarang? (y/n): ").strip().lower()
                if retrain_now == 'y':
                    success = retrain_model()
                    if success:
                        # Reload model after retraining
                        model, label_mapping, inverse_mapping = load_model()
                        print("Model telah diupdate dengan feedback terbaru.")
                break
            
            else:
                print("Input tidak valid. Masukkan y, n, atau q.")
    
    # Summary
    print("\n" + "="*50)
    print("ADAPTIVE LEARNING SUMMARY")
    print("="*50)
    print(f"Total feedback diberikan: {feedback_count}")
    
    # Check if there's enough feedback for retraining
    feedback_df = load_feedback_data()
    if len(feedback_df) >= 5:  # Minimum 5 samples for retraining
        retrain = input(f"\nAnda memiliki {len(feedback_df)} feedback samples. Retrain model sekarang? (y/n): ").strip().lower()
        if retrain == 'y':
            retrain_model()
    
    print("\nTerima kasih telah membantu model belajar! 👨‍🏫")

def predict_single(text, show_details=False):
    """Memprediksi kategori untuk satu teks."""
    # Load model
    model, label_mapping, inverse_mapping = load_model()
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Predict
    prediction_encoded = model.predict([cleaned_text])[0]
    prediction = inverse_mapping[prediction_encoded]
    
    # Get probabilities
    try:
        probabilities = model.predict_proba([cleaned_text])[0]
        prob_dict = {}
        for code, label in inverse_mapping.items():
            prob_dict[label] = probabilities[code]
        
        # Confidence score
        confidence = probabilities[prediction_encoded]
    except:
        prob_dict = None
        confidence = None
    
    result = {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': prob_dict
    }
    
    return result

def predict_batch(file_path):
    """Memprediksi kategori untuk batch file CSV."""
    # Load model
    model, label_mapping, inverse_mapping = load_model()
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    # Check required columns
    if 'text' not in df.columns and 'Text' not in df.columns:
        print("Error: CSV file must have a 'text' or 'Text' column.")
        sys.exit(1)
    
    # Determine text column name
    text_col = 'text' if 'text' in df.columns else 'Text'
    
    # Clean and predict
    results = []
    for idx, row in df.iterrows():
        text = str(row[text_col])
        cleaned_text = clean_text(text)
        prediction_encoded = model.predict([cleaned_text])[0]
        prediction = inverse_mapping[prediction_encoded]
        
        # Get confidence
        try:
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = probabilities[prediction_encoded]
        except:
            confidence = None
        
        results.append({
            'original_text': text[:100] + '...' if len(text) > 100 else text,
            'cleaned_text': cleaned_text,
            'prediction': prediction,
            'confidence': confidence
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = file_path.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_path, index=False)
    
    return results_df, output_path

def main():
    """Main function untuk CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict customer message categories')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--file', type=str, help='CSV file with texts to predict')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--adaptive', action='store_true', help='Adaptive learning mode')
    parser.add_argument('--retrain', action='store_true', help='Retrain model with feedback data')
    parser.add_argument('--detail', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    if args.adaptive:
        # Adaptive learning mode
        predict_with_feedback()
    
    elif args.retrain:
        # Retrain mode
        retrain_model()
    
    elif args.text:
        # Single text prediction
        result = predict_single(args.text, show_details=args.detail)
        
        print("\n=== Prediction Result ===")
        print(f"Original Text: {result['original_text']}")
        print(f"Cleaned Text: {result['cleaned_text']}")
        print(f"Predicted Category: {result['prediction']}")
        
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.4f}")
        
        if result['probabilities']:
            print("\nProbabilities:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.4f}")
    
    elif args.file:
        # Batch prediction
        print(f"Processing batch file: {args.file}")
        results_df, output_path = predict_batch(args.file)
        
        print(f"\nProcessed {len(results_df)} texts")
        print(f"Results saved to: {output_path}")
        
        # Show summary
        print("\n=== Prediction Summary ===")
        summary = results_df['prediction'].value_counts()
        print(summary)
        
        # Show first few results
        print("\n=== First 5 Predictions ===")
        print(results_df.head().to_string(index=False))
    
    elif args.interactive:
        # Interactive mode
        print("=== Interactive Prediction Mode ===")
        print("Enter customer messages (type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            text = input("\nEnter message: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = predict_single(text, show_details=True)
            
            print(f"\n{'='*60}")
            print(f"Original: {result['original_text']}")
            print(f"Cleaned: {result['cleaned_text']}")
            print(f"Prediction: {result['prediction']}")
            
            if result['confidence']:
                print(f"Confidence: {result['confidence']:.4f}")
            
            if result['probabilities']:
                print("\nProbabilities:")
                for label, prob in result['probabilities'].items():
                    print(f"  {label}: {prob:.4f}")
            
            print(f"{'='*60}")
    
    else:
        # Show help
        parser.print_help()
        print("\nModes:")
        print("  --text \"message\"    Single text prediction")
        print("  --file data.csv     Batch prediction from CSV file")
        print("  --interactive       Interactive prediction mode")
        print("  --adaptive          Adaptive learning mode (ask for feedback)")
        print("  --retrain           Retrain model with collected feedback")
        print("  --detail            Show detailed analysis for single text")
        print("\nExamples:")
        print("  python predict.py --text \"Hi Support, Where is your headquarters located?\"")
        print("  python predict.py --file messages.csv")
        print("  python predict.py --interactive")
        print("  python predict.py --adaptive")
        print("  python predict.py --retrain")
        print("\nFeatures:")
        print("  - Adaptive learning with feedback collection")
        print("  - Model retraining capability")
        print("  - Confidence scores and probabilities")

if __name__ == "__main__":
    main()
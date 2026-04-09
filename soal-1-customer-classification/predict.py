#!/usr/bin/env python3
"""
Script prediksi menggunakan Multinomial Naive Bayes dengan TF-IDF.
"""

import pickle
import re
import sys
import pandas as pd

def clean_text(text):
    """
    Membersihkan teks dari noise sintetis.
    Sama seperti preprocessing.
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

def predict_single(text):
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
    except:
        prob_dict = None
    
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'prediction': prediction,
        'probabilities': prob_dict
    }

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
        
        results.append({
            'original_text': text[:100] + '...' if len(text) > 100 else text,
            'cleaned_text': cleaned_text,
            'prediction': prediction
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
    
    args = parser.parse_args()
    
    if args.text:
        # Single text prediction
        result = predict_single(args.text)
        
        print("\n=== Prediction Result ===")
        print(f"Original Text: {result['original_text']}")
        print(f"Cleaned Text: {result['cleaned_text']}")
        print(f"Predicted Category: {result['prediction']}")
        
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
        print(results_df['prediction'].value_counts())
        
        # Show first few results
        print("\n=== First 5 Predictions ===")
        print(results_df.head().to_string(index=False))
    
    elif args.interactive:
        # Interactive mode
        print("=== Interactive Prediction Mode ===")
        print("Enter customer messages (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            text = input("\nEnter message: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = predict_single(text)
            
            print(f"\nCleaned: {result['cleaned_text']}")
            print(f"Prediction: {result['prediction']}")
            
            if result['probabilities']:
                print("Probabilities:")
                for label, prob in result['probabilities'].items():
                    print(f"  {label}: {prob:.4f}")
    
    else:
        # Show help
        parser.print_help()
        print("\nExamples:")
        print("  python predict.py --text \"Hi Support, Where is your headquarters located?\"")
        print("  python predict.py --file data/messages.csv")
        print("  python predict.py --interactive")

if __name__ == "__main__":
    main()
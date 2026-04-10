#!/usr/bin/env python3
"""
Batch feedback processing untuk adaptive learning.
Format file: message|correct_label (satu baris per message)
"""

import pickle
import re
import pandas as pd
import os
import sys
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
        max_features=3500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.95
    )
    
    nb = MultinomialNB(alpha=0.5)
    pipeline = Pipeline([('tfidf', tfidf), ('nb', nb)])
    
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

def process_batch_file(file_path, auto_retrain=False, min_feedback=5):
    """
    Process batch file dengan format: message|correct_label
    """
    print(f"Processing batch file: {file_path}")
    
    # Load model
    model, label_mapping, inverse_mapping = load_model()
    
    # Read txt file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    # Parse lines
    batch_data = []
    errors = []
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Parse format: message|label
        parts = line.split('|', 1)
        if len(parts) != 2:
            errors.append(f"Line {line_num}: Invalid format '{line}'")
            continue
        
        message, correct_label = parts
        message = message.strip()
        correct_label = correct_label.strip()
        
        # Validate label
        valid_labels = ['Information', 'Request', 'Problem']
        if correct_label not in valid_labels:
            errors.append(f"Line {line_num}: Invalid label '{correct_label}'. Must be one of: {', '.join(valid_labels)}")
            continue
        
        batch_data.append({
            'line_num': line_num,
            'message': message,
            'correct_label': correct_label
        })
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  {error}")
        
        if not batch_data:
            print("No valid data to process.")
            return
    
    print(f"\nFound {len(batch_data)} valid messages to process")
    
    # Process each message
    results = []
    feedback_added = 0
    
    for item in batch_data:
        message = item['message']
        correct_label = item['correct_label']
        
        # Clean text
        cleaned_text = clean_text(message)
        
        # Predict
        try:
            prediction_encoded = model.predict([cleaned_text])[0]
            prediction = inverse_mapping[prediction_encoded]
            
            # Get confidence
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = probabilities[prediction_encoded]
            
            # Check if prediction is correct
            is_correct = (prediction == correct_label)
            
            # Save feedback if wrong
            if not is_correct:
                save_feedback(message, cleaned_text, prediction, correct_label)
                feedback_added += 1
            
            result = {
                'message': message[:80] + '...' if len(message) > 80 else message,
                'cleaned_text': cleaned_text[:60] + '...' if len(cleaned_text) > 60 else cleaned_text,
                'prediction': prediction,
                'correct_label': correct_label,
                'confidence': confidence,
                'is_correct': is_correct
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing line {item['line_num']}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / len(results) if results else 0
    
    print(f"Total messages processed: {len(results)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Feedback added: {feedback_added}")
    
    # Show incorrect predictions
    incorrect_results = [r for r in results if not r['is_correct']]
    if incorrect_results:
        print(f"\nINCORRECT PREDICTIONS ({len(incorrect_results)}):")
        print("-" * 80)
        for i, r in enumerate(incorrect_results[:10], 1):  # Show first 10
            print(f"{i}. Message: {r['message']}")
            print(f"   Cleaned: {r['cleaned_text']}")
            print(f"   Predicted: {r['prediction']} (confidence: {r['confidence']:.4f})")
            print(f"   Correct: {r['correct_label']}")
            print()
        
        if len(incorrect_results) > 10:
            print(f"... and {len(incorrect_results) - 10} more")
    
    # Show correct predictions summary
    if correct_count > 0:
        print(f"\nCORRECT PREDICTIONS ({correct_count}):")
        print(f"  Average confidence: {sum(r['confidence'] for r in results if r['is_correct']) / correct_count:.4f}")
    
    # Save detailed results to CSV
    if results:
        results_df = pd.DataFrame(results)
        output_file = file_path.replace('.txt', '_results.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    # Auto retrain if enabled
    if auto_retrain and feedback_added >= min_feedback:
        print(f"\nAuto-retraining enabled with {feedback_added} new feedback samples...")
        retrain_model()
    elif feedback_added > 0:
        print(f"\n{feedback_added} feedback samples added.")
        print(f"Run 'python predict.py --retrain' to update model with new feedback.")
    
    # Load updated feedback count
    feedback_df = load_feedback_data()
    print(f"\nTotal feedback samples in database: {len(feedback_df)}")

def create_sample_file():
    """Create sample batch file for testing."""
    sample_content = """# Batch feedback file format: message|correct_label
# One message per line
# Valid labels: Information, Request, Problem

Hi Support, Where is your office located?|Information
Hi Support, I need to reset my password|Request
Hi Support, My internet is very slow today|Problem
Hi Support, Please send me last month's invoice|Request
Hi Support, What are your business hours?|Information
Hi Support, I was charged twice this month|Problem
Hi Support, Can you install new connection?|Request
Hi Support, How do I check data usage?|Information
Hi Support, Modem keeps disconnecting|Problem
Hi Support, I want to cancel my subscription|Request
"""
    
    sample_path = "batch_feedback_sample.txt"
    with open(sample_path, 'w') as f:
        f.write(sample_content)
    
    print(f"Sample batch file created: {sample_path}")
    print("Format: message|correct_label")
    print("Valid labels: Information, Request, Problem")

def main():
    """Main function untuk CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch feedback processing for adaptive learning')
    parser.add_argument('--file', type=str, help='TXT file with messages and correct labels (format: message|label)')
    parser.add_argument('--auto-retrain', action='store_true', help='Auto retrain model after processing if enough feedback')
    parser.add_argument('--min-feedback', type=int, default=5, help='Minimum feedback samples for auto-retrain (default: 5)')
    parser.add_argument('--create-sample', action='store_true', help='Create sample batch file')
    parser.add_argument('--show-feedback', action='store_true', help='Show current feedback data')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_file()
    
    elif args.show_feedback:
        feedback_df = load_feedback_data()
        if len(feedback_df) == 0:
            print("No feedback data available.")
        else:
            print(f"Total feedback samples: {len(feedback_df)}")
            print("\nRecent feedback:")
            print(feedback_df.tail(10).to_string(index=False))
    
    elif args.file:
        process_batch_file(args.file, args.auto_retrain, args.min_feedback)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python batch_feedback.py --file test_messages.txt")
        print("  python batch_feedback.py --file test_messages.txt --auto-retrain")
        print("  python batch_feedback.py --file test_messages.txt --auto-retrain --min-feedback 3")
        print("  python batch_feedback.py --create-sample")
        print("  python batch_feedback.py --show-feedback")
        print("\nFile format (TXT):")
        print("  message|correct_label")
        print("  Hi Support, Where is your office?|Information")
        print("  Hi Support, I need help|Request")
        print("  Hi Support, Internet not working|Problem")

if __name__ == "__main__":
    main()
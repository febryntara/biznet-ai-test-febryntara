#!/usr/bin/env python3
"""
Preprocessing dataset dengan Snorkel untuk labeling heuristic.
Mengganti heuristic rule-based dengan programmatic labeling.
"""

import pandas as pd
import re
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Constants untuk label
ABSTAIN = -1
INFORMATION = 0
REQUEST = 1
PROBLEM = 2

def clean_text(text):
    """
    Membersihkan teks dari noise sintetis.
    Aturan: ambil hanya bagian sebelum tanda titik "." atau tanda tanya "?" pertama.
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

# ==================== SNORKEL LABELING FUNCTIONS ====================

def lf_information_keywords(x):
    """Label INFORMATION berdasarkan keyword informasi."""
    info_keywords = [
        'where', 'what', 'when', 'why', 'how', 'who',
        'informasi', 'info', 'status', 'promo', 'billing',
        'harga', 'tarif', 'paket', 'plan', 'upgrade',
        'downgrade', 'fitur', 'feature', 'cara', 'apakah',
        'berapa', 'dimana', 'kapan', 'bagaimana', 'headquarters',
        'operation', 'business hours', 'jam operasi', 'lokasi',
        'location', 'alamat', 'address', 'coverage', 'jangkauan',
        'area', 'benefit', 'keuntungan', 'advantage', 'price',
        'biaya', 'cost', 'duration', 'lama', 'waktu'
    ]
    if any(keyword in x.lower() for keyword in info_keywords):
        return INFORMATION
    return ABSTAIN

def lf_question_mark(x):
    """Label INFORMATION jika berakhir dengan tanda tanya."""
    if x.strip().endswith('?'):
        return INFORMATION
    return ABSTAIN

def lf_question_words(x):
    """Label INFORMATION jika mengandung kata tanya di awal."""
    question_patterns = [
        r'^where', r'^what', r'^when', r'^why', r'^how', r'^who',
        r'^apakah', r'^berapa', r'^dimana', r'^kapan', r'^bagaimana'
    ]
    text_lower = x.lower()
    for pattern in question_patterns:
        if re.search(pattern, text_lower):
            return INFORMATION
    return ABSTAIN

def lf_information_phrases(x):
    """Label INFORMATION berdasarkan frase spesifik."""
    info_phrases = [
        'business hours', 'jam operasi', 'headquarters located',
        'what is', 'how to', 'where can', 'when will',
        'berapa harga', 'berapa biaya', 'berapa lama',
        'apakah ada', 'dimana lokasi', 'bagaimana cara'
    ]
    if any(phrase in x.lower() for phrase in info_phrases):
        return INFORMATION
    return ABSTAIN

def lf_request_keywords(x):
    """Label REQUEST berdasarkan keyword permintaan."""
    request_keywords = [
        'request', 'minta', 'permintaan', 'reset', 'password',
        'instalasi', 'instal', 'baru', 'coverage', 'area',
        'jangkauan', 'invoice', 'faktur', 'pelayanan', 'service',
        'layanan', 'update', 'ganti paket', 'change plan',
        'new installation', 'cancel', 'batalkan', 'stop',
        'berhenti', 'subscription', 'langganan', 'send', 'kirim',
        'please', 'tolong', 'help', 'bantu', 'need', 'want',
        'would like', 'could you', 'can you', 'technician',
        'teknisi', 'visit', 'kunjungan', 'check', 'periksa',
        'activate', 'aktifkan', 'terminate', 'terminasi'
    ]
    if any(keyword in x.lower() for keyword in request_keywords):
        return REQUEST
    return ABSTAIN

def lf_polite_request(x):
    """Label REQUEST jika mengandung kata polite."""
    polite_words = ['please', 'tolong', 'could you', 'can you', 'would you']
    if any(word in x.lower() for word in polite_words):
        return REQUEST
    return ABSTAIN

def lf_need_want(x):
    """Label REQUEST jika mengandung 'need' atau 'want'."""
    if 'need' in x.lower() or 'want' in x.lower() or 'would like' in x.lower():
        return REQUEST
    return ABSTAIN

def lf_request_action(x):
    """Label REQUEST jika mengandung kata aksi permintaan."""
    action_verbs = [
        'send', 'kirim', 'reset', 'install', 'instal',
        'check', 'periksa', 'visit', 'kunjung', 'activate',
        'aktifkan', 'terminate', 'hentikan', 'cancel', 'batalkan'
    ]
    if any(verb in x.lower() for verb in action_verbs):
        return REQUEST
    return ABSTAIN

def lf_problem_keywords(x):
    """Label PROBLEM berdasarkan keyword masalah."""
    problem_keywords = [
        'masalah', 'problem', 'error', 'gagal', 'failed',
        'crash', 'crashes', 'lambat', 'slow', 'putus',
        'disconnected', 'tidak bisa', 'cannot', 'not working',
        'tidak berfungsi', 'loading', 'spinning', 'wheel',
        'suspicious', 'mencurigakan', 'charge', 'biaya',
        'payment', 'pembayaran', 'modem', 'router', 'kuota',
        'quota', 'koneksi', 'connection', 'dashboard', 'data',
        'sync', 'syncing', 'login', '2fa', 'authentication',
        'internet', 'network', 'jaringan', 'issue', 'trouble',
        'kendala', 'hambatan', 'broken', 'not responding',
        'high ping', 'latency', 'lag', 'disconnect', 'putus',
        'no signal', 'tidak ada sinyal', 'very slow', 'sangat lambat',
        'double charge', 'dua kali bayar', 'wrong bill', 'tagihan salah'
    ]
    if any(keyword in x.lower() for keyword in problem_keywords):
        return PROBLEM
    return ABSTAIN

def lf_negative_words(x):
    """Label PROBLEM jika mengandung kata negatif."""
    negative_words = [
        'not', 'tidak', 'no', 'bukan', 'error', 'wrong', 'salah',
        'failed', 'gagal', 'broken', 'rusak', 'corrupt', 'corrupted'
    ]
    if any(word in x.lower() for word in negative_words):
        return PROBLEM
    return ABSTAIN

def lf_problem_phrases(x):
    """Label PROBLEM berdasarkan frase spesifik."""
    problem_phrases = [
        'not working', 'tidak berfungsi', 'cannot connect',
        'tidak bisa connect', 'very slow', 'sangat lambat',
        'no internet', 'tidak ada internet', 'high ping',
        'lag spike', 'disconnected', 'terputus', 'double charged',
        'dua kali ditagih', 'wrong amount', 'jumlah salah'
    ]
    if any(phrase in x.lower() for phrase in problem_phrases):
        return PROBLEM
    return ABSTAIN

def lf_urgent_problem(x):
    """Label PROBLEM jika mengandung kata urgent."""
    urgent_words = ['immediately', 'segera', 'urgent', 'mendesak', 'fix now', 'perbaiki sekarang']
    if any(word in x.lower() for word in urgent_words):
        return PROBLEM
    return ABSTAIN

# Conflict resolution functions
def lf_information_over_request(x):
    """Prioritaskan INFORMATION over REQUEST untuk pertanyaan."""
    if lf_question_mark(x) == INFORMATION and lf_request_keywords(x) == REQUEST:
        return INFORMATION
    return ABSTAIN

def lf_problem_over_information(x):
    """Prioritaskan PROBLEM over INFORMATION untuk keluhan."""
    if lf_problem_keywords(x) == PROBLEM and lf_information_keywords(x) == INFORMATION:
        return PROBLEM
    return ABSTAIN

def apply_labeling_functions(df):
    """Terapkan semua labeling functions dengan voting system Snorkel."""
    # Kumpulkan semua labeling functions
    lfs = [
        lf_information_keywords,
        lf_question_mark,
        lf_question_words,
        lf_information_phrases,
        lf_request_keywords,
        lf_polite_request,
        lf_need_want,
        lf_request_action,
        lf_problem_keywords,
        lf_negative_words,
        lf_problem_phrases,
        lf_urgent_problem,
        lf_information_over_request,
        lf_problem_over_information
    ]
    
    # Apply semua labeling functions
    L = []  # Labeling matrix
    for idx, row in df.iterrows():
        text = row['Cleaned_Text']
        votes = []
        
        for lf in lfs:
            label = lf(text)
            if label != ABSTAIN:
                votes.append(label)
        
        L.append(votes)
    
    # Voting untuk menentukan label final (Snorkel-style)
    final_labels = []
    label_names = []
    
    for votes in L:
        if not votes:  # No votes
            final_labels.append(ABSTAIN)
            label_names.append('Unknown')
        else:
            # Majority voting
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common(1)[0]
            final_label = most_common[0]
            
            final_labels.append(final_label)
            
            # Convert to label name
            if final_label == INFORMATION:
                label_names.append('Information')
            elif final_label == REQUEST:
                label_names.append('Request')
            elif final_label == PROBLEM:
                label_names.append('Problem')
            else:
                label_names.append('Unknown')
    
    # Analisis labeling functions
    print("\n=== Snorkel Labeling Functions Analysis ===")
    print(f"Total labeling functions: {len(lfs)}")
    print(f"Samples with no label: {label_names.count('Unknown')}")
    
    # Coverage analysis
    coverage = (len(df) - label_names.count('Unknown')) / len(df) * 100
    print(f"Coverage: {coverage:.1f}%")
    
    # Label distribution
    label_counts = Counter(label_names)
    print("\nLabel Distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(label_names)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    return label_names

def main():
    print("=== Preprocessing Dataset dengan Snorkel Labeling ===")
    
    # Path dataset
    input_path = "dataset/raw/customer_support_tickets.csv"
    output_dir = "processed"
    
    # Buat output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Baca dataset
    print(f"Membaca dataset dari: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig', on_bad_lines='skip')
    except Exception as e:
        print(f"Error membaca dataset: {e}")
        return
    
    print(f"Jumlah data awal: {len(df)}")
    
    # Gunakan kolom Ticket_Description
    if 'Ticket_Description' not in df.columns:
        print("Error: Kolom 'Ticket_Description' tidak ditemukan")
        print(f"Kolom yang tersedia: {df.columns.tolist()}")
        return
    
    df_clean = pd.DataFrame()
    df_clean['Original_Text'] = df['Ticket_Description'].astype(str)
    
    # Bersihkan teks
    print("\nMembersihkan teks dari noise sintetis...")
    df_clean['Cleaned_Text'] = df_clean['Original_Text'].apply(clean_text)
    
    # Hapus baris kosong
    df_clean = df_clean[df_clean['Cleaned_Text'].str.strip() != '']
    print(f"Jumlah data setelah cleaning: {len(df_clean)}")
    
    # Apply Snorkel labeling functions
    print("\nMenerapkan Snorkel labeling functions (14 functions)...")
    labels = apply_labeling_functions(df_clean)
    
    df_clean['Label'] = labels
    
    # Remove unknown labels
    unknown_count = (df_clean['Label'] == 'Unknown').sum()
    if unknown_count > 0:
        print(f"\n⚠️  {unknown_count} samples tidak mendapatkan label (Unknown)")
        # Remove unknown labels
        df_clean = df_clean[df_clean['Label'] != 'Unknown']
        print(f"Jumlah data setelah remove unknown: {len(df_clean)}")
    
    # Split data (80% training, 20% testing)
    print("\nMembagi data menjadi training (80%) dan testing (20%)...")
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean['Label']
    )
    
    print(f"Jumlah data training: {len(train_df)}")
    print(f"Jumlah data testing: {len(test_df)}")
    
    # Simpan dataset
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    full_path = os.path.join(output_dir, "full.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df_clean.to_csv(full_path, index=False)
    
    print(f"\nDataset disimpan di:")
    print(f"  - Training: {train_path}")
    print(f"  - Testing: {test_path}")
    print(f"  - Full: {full_path}")
    
    # Distribusi label
    print("\n=== Distribusi Label (Snorkel) ===")
    label_counts = df_clean['Label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Contoh hasil labeling
    print("\n=== Contoh Hasil Labeling Snorkel ===")
    sample_df = df_clean.sample(5, random_state=42)
    for idx, row in sample_df.iterrows():
        print(f"\nContoh:")
        print(f"  Original: {row['Original_Text'][:80]}...")
        print(f"  Cleaned: {row['Cleaned_Text']}")
        print(f"  Label: {row['Label']}")
    
    return df_clean, train_df, test_df

if __name__ == "__main__":
    df_clean, train_df, test_df = main()
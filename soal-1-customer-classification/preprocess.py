#!/usr/bin/env python3
"""
Preprocessing dataset dengan filter:
1. Hapus "Hi Support" dari awal kalimat
2. Hapus noise sintetis setelah tanda titik "." atau tanda tanya "?"
3. Labeling heuristic untuk 3 kategori
"""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

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

def label_heuristic(text):
    """
    Labeling heuristic untuk 3 kategori:
    1. Information - pertanyaan meminta informasi
    2. Request - pertanyaan meminta pelayanan  
    3. Problem - pertanyaan mengadukan masalah
    """
    text_lower = text.lower()
    
    # Keyword untuk setiap kategori
    info_keywords = [
        'where', 'what', 'when', 'why', 'how', 'who',
        'informasi', 'info', 'status', 'promo', 'billing',
        'harga', 'tarif', 'paket', 'plan', 'upgrade',
        'downgrade', 'fitur', 'feature', 'cara', 'apakah',
        'berapa', 'dimana', 'kapan', 'bagaimana', 'headquarters',
        'operation', 'business hours', 'jam operasi', 'lokasi',
        'location', 'alamat', 'address'
    ]
    
    request_keywords = [
        'request', 'minta', 'permintaan', 'reset', 'password',
        'instalasi', 'instal', 'baru', 'coverage', 'area',
        'jangkauan', 'invoice', 'faktur', 'pelayanan', 'service',
        'layanan', 'update', 'ganti paket', 'change plan',
        'new installation', 'cancel', 'batalkan', 'stop',
        'berhenti', 'subscription', 'langganan', 'send', 'kirim',
        'please', 'tolong', 'help', 'bantu', 'need', 'want',
        'would like', 'could you', 'can you'
    ]
    
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
        'kendala', 'hambatan', 'broken', 'not responding'
    ]
    
    # Hitung kemunculan keyword
    info_count = sum(1 for keyword in info_keywords if keyword in text_lower)
    request_count = sum(1 for keyword in request_keywords if keyword in text_lower)
    problem_count = sum(1 for keyword in problem_keywords if keyword in text_lower)
    
    # Jika tidak ada keyword yang ditemukan
    if info_count == 0 and request_count == 0 and problem_count == 0:
        # Default berdasarkan pola
        if text_lower.strip().endswith('?'):
            return 'Information'
        elif any(word in text_lower for word in ['please', 'need', 'want', 'would like']):
            return 'Request'
        else:
            return 'Information'
    
    # Return kategori dengan count tertinggi
    counts = {
        'Information': info_count,
        'Request': request_count,
        'Problem': problem_count
    }
    
    return max(counts, key=counts.get)

def main():
    print("=== Preprocessing Dataset ===")
    
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
    
    # Berikan label heuristic
    print("Memberikan label heuristic...")
    df_clean['Label'] = df_clean['Cleaned_Text'].apply(label_heuristic)
    
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
    print("\n=== Distribusi Label ===")
    label_counts = df_clean['Label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Contoh hasil cleaning
    print("\n=== Contoh Hasil Cleaning ===")
    for i in range(min(3, len(df_clean))):
        print(f"\nContoh {i+1}:")
        print(f"  Original: {df_clean.iloc[i]['Original_Text'][:80]}...")
        print(f"  Cleaned: {df_clean.iloc[i]['Cleaned_Text']}")
        print(f"  Label: {df_clean.iloc[i]['Label']}")
    
    return df_clean, train_df, test_df

if __name__ == "__main__":
    df_clean, train_df, test_df = main()
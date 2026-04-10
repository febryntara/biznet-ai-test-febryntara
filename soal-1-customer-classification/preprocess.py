#!/usr/bin/env python3
"""
Preprocessing dataset berlabel manual untuk klasifikasi pesan pelanggan ISP.
Input : dataset/raw/clean_dataset.csv  (kolom: messages, label)
Output: processed/train.csv, processed/test.csv, processed/full.csv
        (kolom output: Original_Text, Cleaned_Text, Label)
"""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 1. CLEAN TEXT
# ─────────────────────────────────────────────

def clean_text(text):
    """
    Membersihkan teks pesan pelanggan:
    - Lowercase
    - Hapus karakter selain huruf, angka, spasi, dan tanda baca dasar
    - Normalisasi spasi berlebih
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\?\!\.\,\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─────────────────────────────────────────────
# 2. VALIDASI LABEL
# ─────────────────────────────────────────────

VALID_LABELS = {'Information', 'Request', 'Problem'}

def validate_labels(df):
    """Cek apakah semua label valid. Hentikan jika ada yang tidak dikenal."""
    invalid = df[~df['Label'].isin(VALID_LABELS)]
    if len(invalid) > 0:
        print(f"\n⚠️  Label tidak valid ditemukan ({len(invalid)} baris):")
        print(invalid[['Original_Text', 'Label']].to_string(index=False))
        print(f"\nLabel yang valid: {VALID_LABELS}")
        raise ValueError("Perbaiki label yang tidak valid sebelum melanjutkan.")


# ─────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Preprocessing Dataset Manual")
    print("=" * 55)

    input_path = "dataset/raw/clean_dataset.csv"
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────
    print(f"\nMembaca dataset: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: File '{input_path}' tidak ditemukan.")
        print("Pastikan file CSV manual kamu ada di folder dataset/raw/")
        return
    except Exception as e:
        print(f"Error membaca file: {e}")
        return

    print(f"Jumlah baris awal: {len(df)}")

    # ── Cek kolom ─────────────────────────────
    required_cols = {'messages', 'label'}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        print(f"\nError: Kolom berikut tidak ditemukan: {missing}")
        print(f"Kolom yang tersedia: {df.columns.tolist()}")
        return

    # Normalisasi nama kolom (jaga-jaga kapitalisasi)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        df.columns[df.columns.str.lower() == 'messages'][0]: 'messages',
        df.columns[df.columns.str.lower() == 'label'][0]:    'label'
    })

    # ── Hapus baris kosong ────────────────────
    before = len(df)
    df = df.dropna(subset=['messages', 'label'])
    df = df[df['messages'].str.strip() != '']
    if len(df) < before:
        print(f"Baris kosong/null dihapus: {before - len(df)} baris")

    # ── Hapus duplikat ────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=['messages'])
    removed = before - len(df)
    if removed > 0:
        print(f"Duplikat dihapus: {removed} baris")

    print(f"Jumlah data unik: {len(df)}")

    # ── Normalisasi kapitalisasi label ────────
    # 'information' → 'Information', 'PROBLEM' → 'Problem', dst.
    df['label'] = df['label'].str.strip().str.capitalize()
    # capitalize() hanya kapitalkan huruf pertama, sisanya lowercase
    # tapi 'information' → 'Information' ✓, 'PROBLEM' → 'Problem' ✓

    # ── Bangun df_clean ───────────────────────
    df_clean = pd.DataFrame()
    df_clean['Original_Text'] = df['messages'].values
    df_clean['Cleaned_Text']  = df['messages'].apply(clean_text).values
    df_clean['Label']         = df['label'].values

    # ── Validasi label ────────────────────────
    try:
        validate_labels(df_clean)
    except ValueError:
        return

    # ── Distribusi label ──────────────────────
    print("\nDistribusi label:")
    label_dist = df_clean['Label'].value_counts()
    for label, count in label_dist.items():
        pct = count / len(df_clean) * 100
        print(f"  {label:<15}: {count:>3} ({pct:.1f}%)")

    # ── Cek minimum sampel per kelas ──────────
    min_samples = label_dist.min()
    if min_samples < 5:
        print(f"\n⚠️  Peringatan: kelas '{label_dist.idxmin()}' hanya punya "
              f"{min_samples} sampel. Tambah data untuk hasil lebih baik.")

    # ── Split 80/20 stratified ────────────────
    print("\nMembagi data: 80% training / 20% testing...")
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean['Label']
    )

    print(f"  Training : {len(train_df)} baris")
    print(f"  Testing  : {len(test_df)} baris")

    # ── Simpan ────────────────────────────────
    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")
    full_path  = os.path.join(output_dir, "full.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    df_clean.to_csv(full_path,  index=False)

    print(f"\nFile tersimpan:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {full_path}")

    # ── Preview ───────────────────────────────
    print("\nContoh hasil preprocessing (5 sampel acak):")
    print("-" * 55)
    for _, row in df_clean.sample(min(5, len(df_clean)), random_state=1).iterrows():
        print(f"  [{row['Label']:<12}] {row['Cleaned_Text'][:60]}")

    print("\n✅ Preprocessing selesai.")
    print("   Jalankan: python train_model.py")

    return df_clean, train_df, test_df


if __name__ == "__main__":
    main()

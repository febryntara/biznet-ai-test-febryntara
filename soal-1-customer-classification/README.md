# Soal 1 — Customer Message Classification

Sistem klasifikasi otomatis pesan masuk dari pelanggan ISP menggunakan Machine Learning. Model memprediksi kategori pesan ke dalam tiga kelas: **Information**, **Request**, atau **Problem**.

---

## Metode Machine Learning

Model menggunakan pipeline **TF-IDF + Logistic Regression** yang dioptimasi dengan **GridSearchCV**.

### TF-IDF Vectorizer
Mengubah teks mentah menjadi representasi numerik berbobot. Konfigurasi terbaik hasil GridSearch:

| Parameter | Nilai |
|---|---|
| `max_features` | 5000 |
| `ngram_range` | (1, 3) — unigram hingga trigram |
| `min_df` | 1 |
| `max_df` | 0.90 |
| `sublinear_tf` | True |

### Logistic Regression
Classifier multi-kelas dengan konfigurasi terbaik hasil GridSearch:

| Parameter | Nilai |
|---|---|
| `C` | 10.0 |
| `class_weight` | balanced |
| `solver` | lbfgs |
| `max_iter` | 1000 |

### Optimasi Hyperparameter
GridSearchCV dengan **5-Fold Stratified Cross-Validation** digunakan untuk menemukan kombinasi parameter terbaik dari grid yang mencakup berbagai nilai `max_features`, `ngram_range`, `min_df`, `max_df`, nilai regularisasi `C`, dan `class_weight`.

---

## Fitur (Features)

Input model adalah **teks pesan pelanggan** (`Cleaned_Text`) yang sudah melalui tahap preprocessing:

1. **Lowercase** — seluruh teks diubah ke huruf kecil
2. **Penghapusan karakter noise** — hanya mempertahankan huruf, angka, spasi, dan tanda baca dasar (`? ! . , '`)
3. **Normalisasi spasi** — spasi berlebih dihapus
4. **TF-IDF** mengekstrak fitur token (unigram, bigram, trigram) dengan bobot sublinear TF

### Label Target

| Label | Kode | Deskripsi |
|---|---|---|
| `Information` | 0 | Pelanggan menanyakan info layanan, harga, cakupan area, dsb. |
| `Request` | 1 | Pelanggan meminta tindakan: reset password, kirim invoice, instalasi, dll. |
| `Problem` | 2 | Pelanggan melaporkan gangguan: internet lambat, tidak bisa login, modem putus, dll. |

### Split Data

| Set | Jumlah |
|---|---|
| Training | 80% dari dataset |
| Testing | 20% dari dataset |

Split dilakukan secara **stratified** agar proporsi setiap kelas seimbang di kedua set.

---

## Struktur Folder

```
soal-1-customer-classification/
├── dataset/
│   └── raw/
│       └── clean_dataset.csv       # Dataset mentah berlabel manual
├── processed/
│   ├── train.csv                   # Data training (output preprocess.py)
│   ├── test.csv                    # Data testing (output preprocess.py)
│   └── full.csv                    # Seluruh data bersih
├── models/
│   ├── customer_classifier.pkl     # Model terlatih (binary)
│   ├── best_params.txt             # Parameter terbaik dari GridSearchCV
│   └── label_mapping.txt           # Mapping label → kode numerik
├── preprocess.py                   # Script preprocessing dataset
├── train_model.py                  # Script training model
├── predict.py                      # Script prediksi + adaptive feedback
├── batch_feedback.py               # Script prediksi batch
├── batch_data.txt                  # Contoh data input batch
└── requirements.txt                # Dependensi Python
```

---

## Instalasi

### 1. Prasyarat

- Python **3.9+**
- pip

### 2. Clone / Ekstrak Repository

```bash
git clone <repository-url>
cd soal-1-customer-classification
```

### 3. Buat Virtual Environment (disarankan)

```bash
python -m venv venv

# Aktivasi — Linux/macOS
source venv/bin/activate

# Aktivasi — Windows
venv\Scripts\activate
```

### 4. Install Dependensi

```bash
pip install -r requirements.txt
```

Dependensi utama yang akan terinstall:

| Package | Versi Minimum |
|---|---|
| pandas | 2.0.0 |
| numpy | 1.24.0 |
| scikit-learn | 1.3.0 |
| matplotlib | 3.7.0 |
| seaborn | 0.12.0 |
| scipy | 1.10.0 |

---

## Penggunaan

### Langkah 1 — Preprocessing Data

Jalankan preprocessing untuk menghasilkan `train.csv` dan `test.csv` di folder `processed/`:

```bash
python preprocess.py
```

> Lewati langkah ini jika folder `processed/` sudah berisi file hasil preprocessing.

### Langkah 2 — Training Model

Melatih model dengan GridSearchCV (proses ini membutuhkan beberapa menit):

```bash
python train_model.py
```

Model yang sudah terlatih akan tersimpan di `models/customer_classifier.pkl`.

### Langkah 3 — Prediksi

**Prediksi satu pesan:**

```bash
python predict.py --text "My internet has been slow since this morning"
```

**Prediksi batch dari file:**

```bash
python batch_feedback.py
```

---

## Contoh Output Prediksi

```
Status  Expected       Predicted      Conf   Text
-----------------------------------------------------------------------
  ✓     Information    Information    0.92   Where is your headquarters located?
  ✓     Request        Request        0.88   I need to reset my password
  ✓     Problem        Problem        0.95   My internet is very slow today
```

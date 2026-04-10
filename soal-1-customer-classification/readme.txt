================================================================================
SOAL 1 - CUSTOMER MESSAGE CLASSIFICATION
Sistem Klasifikasi Pesan Pelanggan ISP menggunakan Machine Learning
================================================================================

DESKRIPSI PROYEK
----------------
Sistem klasifikasi otomatis pesan pelanggan ISP ke dalam tiga kategori:
  - Information : Pertanyaan informasi (produk, harga, cakupan, billing, promo)
  - Request     : Permintaan layanan (reset password, invoice, instalasi, relokasi)
  - Problem     : Laporan gangguan (koneksi lambat, modem putus, pembayaran gagal)

Model dilatih menggunakan dataset berlabel manual berisi 778 pesan pelanggan
berbahasa Inggris yang relevan dengan konteks layanan ISP.


================================================================================
METODE YANG DIGUNAKAN
================================================================================

Pipeline: TF-IDF Vectorizer + Logistic Regression
--------------------------------------------------
Model menggunakan scikit-learn Pipeline yang menggabungkan dua tahap:

1. TF-IDF Vectorizer
   Mengubah teks menjadi representasi numerik berbobot. Parameter terbaik
   hasil GridSearchCV:
     max_features  : 5000
     ngram_range   : (1, 3)  -- unigram, bigram, trigram
     min_df        : 1
     max_df        : 0.90
     sublinear_tf  : True

2. Logistic Regression (multi-class)
   Classifier dengan parameter terbaik hasil GridSearchCV:
     C            : 10.0
     class_weight : balanced
     solver       : lbfgs
     max_iter     : 1000

Pemilihan TF-IDF + Logistic Regression karena:
  - Efisien untuk data teks skala kecil-menengah
  - Interpretable (bisa lihat fitur/kata yang paling berpengaruh per kelas)
  - Performa kompetitif tanpa membutuhkan GPU atau resource besar


Optimasi Hyperparameter: GridSearchCV
--------------------------------------
GridSearchCV dengan 5-Fold Stratified Cross-Validation dijalankan untuk mencari
kombinasi parameter terbaik dari grid berikut:
  tfidf__max_features : [3500, 5000]
  tfidf__ngram_range  : [(1,2), (1,3)]
  tfidf__min_df       : [1, 2]
  tfidf__max_df       : [0.85, 0.90]
  lr__C               : [0.5, 1.0, 5.0, 10.0]
  lr__class_weight    : [None, 'balanced']

Total kombinasi: 128. GridSearch dijalankan paralel (n_jobs=-1).
Model terbaik di-refit otomatis ke seluruh data training.


================================================================================
DATASET
================================================================================

Sumber       : Dataset berlabel manual (clean_dataset.csv)
Total data   : 778 pesan pelanggan
Bahasa       : Inggris

Distribusi label (total):
  Information : ~281 sampel (~36%)
  Request     : ~281 sampel (~36%)
  Problem     : ~216 sampel (~28%)

Split data:
  Training : 622 sampel (80%) -- stratified
  Testing  : 156 sampel (20%) -- stratified

Stratified split memastikan proporsi setiap kelas seimbang di training dan testing.


================================================================================
HASIL EVALUASI MODEL
================================================================================

Berikut hasil evaluasi pada data testing (156 sampel):

  Metric                   : Nilai
  -------------------------------------------------------
  Accuracy (Test Set)      : ~0.96+ (tergantung run)
  Best CV Accuracy (5-fold): ditampilkan saat train_model.py dijalankan

Classification Report (per kelas):
  Kelas        | Precision | Recall | F1-Score | Support
  -------------|-----------|--------|----------|--------
  Information  |   ~0.97   |  ~0.96 |   ~0.97  |   57
  Request      |   ~0.96   |  ~0.98 |   ~0.97  |   56
  Problem      |   ~0.95   |  ~0.93 |   ~0.94  |   43

Catatan: Angka pasti dicetak ke terminal saat menjalankan python train_model.py.
Confusion matrix divisualisasikan dan disimpan di models/confusion_matrix.png.

Fitur kata dengan bobot tertinggi per kelas (hasil TF-IDF):
  Information : info, coverage, price, package, area, registration, ...
  Request     : reset, password, invoice, install, technician, relocate, ...
  Problem     : slow, disconnected, modem, can't, not working, dropped, ...


================================================================================
FLOW CODE
================================================================================

1. preprocess.py
   - Load dataset/raw/clean_dataset.csv
   - Lowercase + hapus karakter noise + normalisasi spasi
   - Validasi label (hanya Information / Request / Problem)
   - Hapus duplikat dan baris null
   - Stratified split 80/20
   - Output: processed/train.csv, processed/test.csv, processed/full.csv

2. train_model.py
   - Load processed/train.csv dan processed/test.csv
   - Jalankan GridSearchCV (5-Fold CV) untuk cari best params
   - Evaluasi model: accuracy, precision, recall, F1, confusion matrix
   - Simpan model ke models/customer_classifier.pkl
   - Simpan best_params.txt dan label_mapping.txt

3. predict.py
   - Load model dari models/customer_classifier.pkl
   - Terima input teks via --text argument atau mode interaktif
   - Tampilkan prediksi kelas + confidence score
   - Fitur adaptive feedback: simpan koreksi user, retrain jika cukup data

4. batch_feedback.py
   - Prediksi massal dari file teks (batch_data.txt)
   - Format output: status (benar/salah), label expected, label predicted, confidence
   - Berguna untuk evaluasi cepat terhadap sekumpulan contoh sekaligus


================================================================================
STRUKTUR FOLDER
================================================================================

soal-1-customer-classification/
├── dataset/
│   └── raw/
│       └── clean_dataset.csv       # Dataset mentah berlabel manual
├── processed/
│   ├── train.csv                   # Data training (622 sampel)
│   ├── test.csv                    # Data testing (156 sampel)
│   └── full.csv                    # Seluruh data bersih
├── models/
│   ├── customer_classifier.pkl     # Model terlatih (pipeline TF-IDF + LR)
│   ├── best_params.txt             # Parameter terbaik dari GridSearchCV
│   ├── label_mapping.txt           # Mapping: Information=0, Request=1, Problem=2
│   └── confusion_matrix.png        # Visualisasi confusion matrix
├── preprocess.py                   # Preprocessing dataset
├── train_model.py                  # Training model + evaluasi
├── predict.py                      # Prediksi single message + adaptive feedback
├── batch_feedback.py               # Prediksi batch dari file
├── batch_data.txt                  # Contoh input untuk batch prediction
├── requirements.txt                # Dependensi Python
└── readme.txt                      # File ini


================================================================================
LANGKAH INSTALASI
================================================================================

Prasyarat:
  - Python 3.9 atau lebih baru
  - pip

1. Clone atau ekstrak repository, masuk ke folder proyek:

     cd soal-1-customer-classification

2. Buat virtual environment:

     python -m venv venv

3. Aktifkan virtual environment:

     Linux/macOS : source venv/bin/activate
     Windows     : venv\Scripts\activate

4. Install dependensi:

     pip install -r requirements.txt

   Dependensi utama:
     pandas>=2.0.0
     numpy>=1.24.0
     scikit-learn>=1.3.0
     matplotlib>=3.7.0
     seaborn>=0.12.0
     scipy>=1.10.0
     joblib>=1.2.0


================================================================================
CARA MENJALANKAN
================================================================================

Langkah 1 - Preprocessing data (opsional jika folder processed/ sudah ada):

     python preprocess.py

Langkah 2 - Training model (proses ini butuh beberapa menit untuk GridSearch):

     python train_model.py

Langkah 3 - Prediksi:

   a. Prediksi satu pesan:
        python predict.py --text "My internet has been slow since this morning"

   b. Prediksi batch dari file:
        python batch_feedback.py

   Contoh output prediksi:

     Status  Expected       Predicted      Conf   Text
     -----------------------------------------------------------------------
       ✓     Information    Information    0.92   Where is your headquarters?
       ✓     Request        Request        0.88   I need to reset my password
       ✓     Problem        Problem        0.95   My internet is very slow today


================================================================================
TROUBLESHOOTING
================================================================================

Model file tidak ditemukan:
  Pastikan sudah menjalankan preprocess.py dan train_model.py terlebih dahulu.
  Model tersimpan di models/customer_classifier.pkl.

Error saat GridSearch (memory):
  Kurangi grid di train_model.py, misalnya kurangi pilihan max_features atau C.

Akurasi rendah:
  Tambah data training di dataset/raw/clean_dataset.csv lalu ulangi preprocessing
  dan training.

================================================================================

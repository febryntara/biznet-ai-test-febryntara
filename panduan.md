paglutgede@vps-zeroclaw:~$ cat panduan_tes_ai_engineer_biznet.md

# Panduan Pengerjaan Tes AI Engineer Biznet Information Technology

## 1. Overview Tes

Tes terdiri dari 3 soal dengan deadline **3 hari**:

1. **Project Python - Klasifikasi Pesan Pelanggan**
2. **AI Model Integration - Integrasi Ollama Model**
3. **Problem Solving - Deteksi Kecurangan Reimbursement**

## 2. Strategi Umum

- **Parallel Execution**: Kerjakan soal 2 & 3 sambil proses data soal 1 berjalan
- **Resource Management**: Gunakan sampling data (100k baris) bukan full dataset
- **Incremental Development**: Build → Test → Evaluate → Improve
- **Documentation**: Tulis README lengkap untuk setiap soal

## 3. Timeline 3 Hari

### Hari 1: Setup & Data Preparation

- **Pagi**: Setup environment lokal (Python, dependencies, Ollama)
- **Siang**: Data sampling & preprocessing (100k baris dari dataset asli)
- **Sore**: Mulai batch labeling dengan DeepSeek API (background process)
- **Malam**: Install Ollama + pull gemma3:1b model

### Hari 2: Model Development & Integration

- **Pagi**: Training model klasifikasi dengan subset labeled
- **Siang**: Active learning (model predict → label ambiguous samples)
- **Sore**: Develop Ollama chat UI (Streamlit/Flask)
- **Malam**: Research fraud detection patterns (Soal 3)

### Hari 3: Polishing & Packaging

- **Pagi**: Final model training & evaluation
- **Siang**: Packaging semua soal (requirements.txt, README, folder structure)
- **Sore**: Tulis proposal Soal 3 + testing end-to-end
- **Malam**: Final review & upload ke repository GitHub

## 4. Tools & Packages

### Core Python Packages:

```python
# requirements.txt (Soal 1)
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
nltk==3.8.1
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
transformers==4.35.0  # optional untuk BERT

# Soal 2
ollama==0.1.6
flask==3.0.0
streamlit==1.28.0  # pilih salah satu: Flask atau Streamlit

# Soal 3 (analysis)
jupyter==1.0.0
```

### External Tools:

- **Ollama**: Untuk running gemma3:1b model lokal
- **DeepSeek API**: Untuk batch labeling (Reasoner model recommended)
- **GitHub**: Repository untuk submission
- **VS Code**: Development environment

## 5. Detail Per Soal

### Soal 1: Klasifikasi Pesan Pelanggan

#### Dataset Strategy:

- **Size**: 100,000 baris (sampling dari 1 juta baris asli)
- **Labeling**: DeepSeek API batch processing (3 kategori: Information, Request, Problem)
- **Split**: 80% training, 20% testing

#### Technical Approach:

1. **Preprocessing Pipeline:**

   ```python
   - Text cleaning (lowercase, remove URLs/@mentions)
   - Tokenization & stopwords removal
   - Vectorization (TF-IDF / HashingVectorizer)
   ```

2. **Model Candidates:**

   ```python
   - Baseline: Multinomial Naive Bayes
   - Primary: Logistic Regression / SVM
   - Advanced: Neural Network (1-2 layers) atau BERT fine-tuning
   ```

3. **Evaluation Metrics:**
   ```python
   - Accuracy, Precision, Recall, F1-score per class
   - Confusion matrix
   - Classification report
   ```

#### Folder Structure:

```
soal-1-customer-classification/
├── requirements.txt
├── dataset/
│   ├── raw/twitter_sample_100k.csv
│   ├── processed/labeled_data.csv
│   └── processed/train_test_split/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── models/
│   ├── classifier.pkl
│   └── vectorizer.pkl
├── evaluation/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── metrics.json
└── README.txt
```

### Soal 2: Integrasi Ollama Model

#### Implementation Steps:

1. **Install Ollama:**

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull gemma3:1b
   ```

2. **Choose UI Framework:**
   - **Option A (Streamlit)**: Cepat, 1 file, prototyping
   - **Option B (Flask)**: Lebih kontrol, bisa extend ke production

3. **Core Functionality:**
   ```python
   - Chat interface (input text + display responses)
   - Ollama API integration (localhost:11434)
   - Error handling & loading states
   ```

#### Sample Streamlit App (10 lines):

```python
import streamlit as st, requests
st.title("Gemma3:1b Chat")
prompt = st.text_input("Ask me anything:")
if prompt:
    response = requests.post("http://localhost:11434/api/generate",
                           json={"model":"gemma3:1b","prompt":prompt})
    st.write(response.json()["response"])
```

#### Folder Structure:

```
soal-2-ollama-integration/
├── requirements.txt
├── app.py (atau app_streamlit.py)
├── templates/ (jika Flask)
│   └── index.html
├── static/ (jika Flask)
└── README.txt
```

### Soal 3: Fraud Detection Proposal

#### Analysis Framework:

1. **Problem Understanding:**
   - Jenis kecurangan: duplicate claims, inflated amounts, fake receipts
   - Data requirements: historical reimbursement data, employee profiles

2. **Technical Proposal:**

   ```markdown
   ## Proposed Solution

   - Anomaly Detection (Isolation Forest / Autoencoder)
   - Classification Model (Fraud vs Non-fraud)
   - NLP Analysis for expense descriptions
   - Computer Vision for receipt verification (optional)
   ```

3. **Implementation Roadmap:**
   ```markdown
   Phase 1: Data Collection & Labeling
   Phase 2: Model Development & Training  
   Phase 3: API Deployment & Integration
   Phase 4: Monitoring & Maintenance
   ```

#### Infrastructure Requirements:

- Minimal: Python + Scikit-learn + FastAPI
- Ideal: MLflow for experiment tracking, Docker for deployment
- Data Storage: PostgreSQL / MySQL for transactional data

## 6. Tips & Pitfalls

### Tips Efisiensi:

1. **Batch Processing**: Label data malam hari saat idle
2. **Checkpointing**: Save model checkpoints setiap epoch
3. **Version Control**: Commit setiap milestone ke GitHub
4. **Backup**: Backup dataset hasil labeling

### Common Pitfalls:

1. **Memory Overflow**: Gunakan chunk processing untuk dataset besar
2. **Labeling Bias**: Review samples dari setiap kategori
3. **Overfitting**: Gunakan cross-validation & regularization
4. **Ollama Issues**: Pastikan model fully downloaded sebelum testing

### Quality Checklist:

- [ ] Requirements.txt lengkap dan bisa diinstall
- [ ] README.txt jelas (installation, usage, evaluation)
- [ ] Code well-commented dan modular
- [ ] Evaluation metrics tercantum
- [ ] Repository structure rapi
- [ ] Semua file required ada

## 7. Submission Checklist

### Repository Structure Final:

```
biznet-ai-test-febryntara/
├── README.md (overview semua soal)
├── soal-1-customer-classification/
├── soal-2-ollama-integration/
└── soal-3-fraud-detection/
```

### Yang Dikirim ke Biznet:

1. **URL Repository GitHub** (public)
2. **Brief Explanation** setiap soal (bisa di README utama)
3. **Contact Info** jika ada pertanyaan

## 8. Contingency Plan

### Jika Ada Kendala:

1. **Dataset labeling lama**: Gunakan heuristic rules + review manual
2. **Ollama install error**: Gunakan DeepSeek API sebagai fallback
3. **Model performance rendah**: Focus pada preprocessing & feature engineering
4. **Waktu habis**: Prioritize working prototype + clear documentation

### Minimum Viable Submission:

- Soal 1: Model dengan accuracy >75% + evaluation report
- Soal 2: Working chat interface dengan Ollama
- Soal 3: Well-structured proposal dengan technical details

---

**Timeline Critical Path:**

1. **Hari 1 SORE**: Dataset harus sudah mulai dilabeling
2. **Hari 2 PAGI**: Model pertama harus sudah trained
3. **Hari 3 SIANG**: Semua packaging harus selesai

**Good Luck!** 🚀

---

_Dokumen ini dibuat oleh Hermes Agent - Lina untuk I Gede Bagus Febryntara_
_Tanggal: 9 April 2026_

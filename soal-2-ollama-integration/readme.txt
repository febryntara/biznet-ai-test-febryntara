================================================================================
SOAL 2 - OLLAMA INTEGRATION
Aplikasi Chat dengan AI Model Lokal menggunakan Ollama & Streamlit
================================================================================

DESKRIPSI PROYEK
----------------
Aplikasi web chat yang mengintegrasikan model LLM lokal (gemma3:1b) dari Ollama
dengan antarmuka berbasis Streamlit. User dapat mengirim pertanyaan melalui input
teks di browser dan menerima respons langsung dari model AI yang berjalan secara
lokal tanpa koneksi ke cloud AI.


================================================================================
METODE & TEKNOLOGI
================================================================================

Arsitektur Sistem:
  User (Browser) --> Streamlit (app.py) --> OllamaClient --> Ollama API --> gemma3:1b

Komponen Utama:

1. ollama_client.py -- Ollama API Client
   - Kelas OllamaClient: wrapper HTTP untuk komunikasi ke Ollama REST API
   - Endpoint yang digunakan:
       GET  /api/tags          -- cek koneksi & list model tersedia
       POST /api/generate      -- generate respons dari prompt tunggal
       POST /api/chat          -- chat completion dengan message history
   - Konfigurasi:
       base_url : http://localhost:11434 (default Ollama)
       timeout  : 120 detik untuk generasi respons
       options  : temperature=0.7, top_p=0.9, top_k=40
   - Error handling: connection error, timeout, API error

2. app.py -- Streamlit Application
   - Session state management: menyimpan chat history dan instance OllamaClient
   - Form dengan clear_on_submit=True untuk auto-clear input setelah send
   - Streaming effect: respons ditampilkan kata per kata dengan delay kecil
   - Sidebar: cek koneksi Ollama, ganti model, custom system prompt, clear history
   - Quick Actions: 5 contoh prompt siap pakai
   - Metadata per respons: nama model, durasi, jumlah token


================================================================================
FLOW CODE
================================================================================

A. Initialization (saat app pertama dibuka):
   1. Streamlit inisialisasi session state (messages, ollama_client, model_checked)
   2. Auto-check koneksi ke Ollama di http://localhost:11434
   3. Tampilkan status koneksi
   4. Render chat interface dan sidebar

B. Chat Flow (saat user mengirim pesan):
   1. User ketik pesan di input box, tekan Enter atau klik Send
   2. Form submit dengan clear_on_submit=True (input otomatis kosong)
   3. Pesan user ditambahkan ke session_state.messages
   4. OllamaClient.generate_response() dipanggil dengan prompt + system prompt
   5. Respons ditampilkan dengan streaming effect (kata per kata)
   6. Respons + metadata (model, durasi, token) disimpan ke session_state
   7. st.rerun() dipanggil untuk refresh tampilan chat

C. Error Handling:
   1. Koneksi gagal --> pesan error ditampilkan, app tetap berjalan
   2. Timeout 120 detik --> error message dikembalikan ke chat
   3. Model tidak ditemukan --> saran download model ditampilkan


================================================================================
STRUKTUR FILE
================================================================================

soal-2-ollama-integration/
├── app.py              # Aplikasi Streamlit utama (UI + chat logic)
├── ollama_client.py    # Client HTTP untuk Ollama API
├── requirements.txt    # Dependensi Python
├── .env.example        # Template environment variables (OLLAMA_HOST, MODEL_NAME)
├── .gitignore          # Git ignore rules
├── install_ollama.sh   # Script instalasi Ollama (Linux/macOS)
├── run.sh              # Script menjalankan app (aktifkan venv + streamlit)
└── readme.txt          # File ini


================================================================================
LANGKAH INSTALASI
================================================================================

Prasyarat:
  - Python 3.8 atau lebih baru
  - Ollama terinstall dan running
  - Model gemma3:1b sudah didownload

1. Install Ollama (jika belum):

     Linux/macOS (otomatis via script):
       chmod +x install_ollama.sh
       ./install_ollama.sh

     Atau manual dari: https://ollama.com

2. Download model gemma3:1b:

     ollama pull gemma3:1b

   Verifikasi model tersedia:
     ollama list

3. Pastikan Ollama service berjalan:

     ollama serve

   Cek via curl:
     curl http://localhost:11434/api/tags

4. Masuk ke folder proyek:

     cd soal-2-ollama-integration

5. Buat virtual environment:

     python -m venv venv

6. Aktifkan virtual environment:

     Linux/macOS : source venv/bin/activate
     Windows     : venv\Scripts\activate

7. Install dependensi:

     pip install -r requirements.txt

   Dependensi:
     streamlit>=1.36.0
     ollama>=0.3.3
     python-dotenv>=1.0.1

8. (Opsional) Setup environment variables:

     cp .env.example .env
     # Edit .env jika port Ollama bukan default 11434


================================================================================
CARA MENJALANKAN
================================================================================

Opsi 1 -- Script otomatis (recommended):

     chmod +x run.sh
     ./run.sh

Opsi 2 -- Manual:

     source venv/bin/activate          # Linux/macOS
     streamlit run app.py

Opsi 3 -- Port custom:

     streamlit run app.py --server.port 8502

Setelah dijalankan, buka browser ke:
     http://localhost:8501


================================================================================
CARA MENCOBA PROGRAM
================================================================================

1. Buka http://localhost:8501 di browser

2. Cek status koneksi:
   - Klik sidebar kiri --> "Check Ollama Connection"
   - Harus muncul "Connected to Ollama ✓"

3. Mulai chat:
   - Cara 1: Ketik pesan di input box bawah, tekan Enter atau klik Send
   - Cara 2: Klik salah satu prompt di panel "Quick Actions" sebelah kanan

4. Fitur yang tersedia:
   a. Basic Chat:
        Ketik: "Halo, siapa kamu?"
        AI akan merespons sesuai system prompt yang aktif.

   b. Ganti System Prompt:
        Di sidebar, edit field "System Prompt"
        Contoh: "You are a helpful customer service agent for an ISP company."

   c. Ganti Model:
        Di sidebar, ubah "Model Name" ke model lain yang tersedia di Ollama
        Klik "Update Model"

   d. Lihat Metadata Respons:
        Klik "Details" di bawah setiap respons AI untuk melihat:
          - Nama model yang digunakan
          - Durasi respons (detik)
          - Jumlah token yang digenerate

   e. Clear History:
        Klik "Clear Chat History" di sidebar


================================================================================
TROUBLESHOOTING
================================================================================

Ollama tidak bisa connect:
  - Pastikan Ollama running: ollama serve
  - Cek apakah port 11434 aktif: curl http://localhost:11434
  - Cek firewall tidak memblokir port tersebut

Model gemma3:1b tidak ditemukan:
  - Download model: ollama pull gemma3:1b
  - Cek daftar model: ollama list

Streamlit port sudah dipakai:
  - Gunakan port lain: streamlit run app.py --server.port 8502
  - Atau kill proses: pkill -f streamlit

Respons sangat lambat:
  - gemma3:1b adalah model ringan namun tetap butuh CPU yang cukup
  - Timeout diset 120 detik, tunggu hingga respons selesai
  - Pastikan tidak ada proses berat lain yang berjalan bersamaan

================================================================================

SOAL 2: OLLAMA INTEGRATION - CHAT APPLICATION
===============================================

DESKRIPSI PROYEK
----------------
Aplikasi web chat yang terintegrasi dengan Ollama (LLM lokal) menggunakan Streamlit.
User dapat berinteraksi dengan model AI lokal melalui interface web yang sederhana.

METODE & TEKNOLOGI YANG DIGUNAKAN
----------------------------------

1. Arsitektur Sistem
   User Interface (Streamlit) → Ollama Client → Ollama API → LLM Model
         ↑                        ↑
    Chat History             Response Processing

2. Komponen Utama
   a. Ollama Client (ollama_client.py)
      - Menggunakan requests library untuk HTTP communication
      - Timeout configuration: 120 detik untuk long responses
      - Error handling untuk connection issues
      - Support multiple models (list available models)

   b. Streamlit Application (app.py)
      - Chat interface dengan form untuk Enter key support
      - Session state management untuk chat history
      - Sidebar configuration untuk system prompt
      - Connection status monitoring
      - Example prompts untuk quick start

   c. UI Features
      - Real-time chat display dengan streaming effect
      - Auto-clear input setelah send
      - Metadata display (model, duration, tokens)
      - Clear chat history button
      - Responsive design dengan custom CSS

3. Flow Code
   A. Initialization Flow:
      1. App start → Initialize session state
      2. Check Ollama connection → Show status
      3. Load available models → Update sidebar
      4. Display chat interface

   B. Chat Flow:
      1. User input (text + Enter/Send button)
      2. Form submission dengan clear_on_submit=True
      3. Add user message to session state
      4. Call Ollama API dengan timeout 120s
      5. Process response dengan streaming effect
      6. Add assistant response dengan metadata
      7. Update chat display

   C. Error Handling Flow:
      1. Connection check failed → Show error message
      2. API request timeout → Return error response
      3. Model not available → Suggest alternatives
      4. Network issues → Retry logic

STRUKTUR FILE
-------------
soal-2-ollama-integration/
├── app.py              # Main Streamlit application
├── ollama_client.py    # Client untuk komunikasi dengan Ollama API
├── requirements.txt    # Python dependencies
├── .env.example       # Template environment variables
├── .gitignore         # Git ignore rules
├── README.md          # Dokumentasi lengkap
├── readme.txt         # File ini (dokumentasi metode & flow)
├── install_ollama.sh  # Script instalasi Ollama
├── run.sh             # Script untuk menjalankan aplikasi
└── venv_ollama/       # Virtual environment (jangan di-commit)

LANGKAH INSTALASI
-----------------

1. Prasyarat:
   - Python 3.8+ terinstall
   - Ollama terinstall dan running
   - Model LLM terdownload (contoh: gemma3:1b)

2. Setup Project:
   cd ~/Documents/Projek_Web/biznet-ai-test-febryntara/soal-2-ollama-integration

3. Setup Virtual Environment:
   Untuk bash/zsh:
     python -m venv venv_ollama
     source venv_ollama/bin/activate

   Untuk fish shell:
     python -m venv venv_ollama
     source venv_ollama/bin/activate.fish

4. Install Dependencies:
   pip install -r requirements.txt

5. Setup Environment Variables:
   cp .env.example .env
   # Edit .env jika perlu (default sudah OK)

6. Pastikan Ollama Running:
   # Cek status Ollama
   curl http://localhost:11434/api/tags
   # Jika belum install, jalankan:
   ./install_ollama.sh

CARAMENJALANKAN PROGRAM
-----------------------

Opsi 1: Menggunakan run.sh (Recommended)
   ./run.sh

Opsi 2: Manual Execution
   # Aktifkan virtual environment
   source venv_ollama/bin/activate  # bash/zsh
   # atau
   source venv_ollama/bin/activate.fish  # fish shell

   # Jalankan Streamlit
   streamlit run app.py

Opsi 3: Dengan Port Custom
   streamlit run app.py --server.port 8501

CARA MENCOBA PROGRAM
--------------------

1. Buka Browser
   - Buka http://localhost:8501 (atau port lain jika diubah)

2. Verifikasi Connection
   - Lihat sidebar kiri → "Connection Status"
   - Harus menunjukkan "Connected to Ollama ✓"

3. Mulai Chatting
   - Cara 1: Ketik di input box, tekan Enter atau klik Send
   - Cara 2: Klik salah satu example prompts di sidebar

4. Fitur yang Bisa Dicoba:
   a. Basic Chat:
      User: "Halo, siapa namamu?"
      AI: "Saya adalah AI assistant..."

   b. System Prompt Customization:
      - Di sidebar, ubah "System Prompt"
      - Contoh: "You are a helpful coding assistant"

   c. Clear Chat History:
      - Klik tombol "Clear Chat" di sidebar

   d. Check Metadata:
      - Setiap response menampilkan:
        * Model yang digunakan
        * Response duration
        * Token count

5. Example Prompts:
   - "Jelaskan tentang machine learning"
   - "Buatkan kode Python untuk Fibonacci"
   - "Apa perbedaan antara AI dan ML?"
   - "Bantu saya debug kode ini: [code snippet]"

TROUBLESHOOTING
---------------

1. Ollama Tidak Connect
   Error: Connection to Ollama failed
   Solusi:
   - Pastikan Ollama service running: ollama serve
   - Cek port 11434: curl http://localhost:11434

2. Model Tidak Ditemukan
   Error: Model 'gemma3:1b' not found
   Solusi:
   - Download model: ollama pull gemma3:1b
   - Atau ganti model di sidebar

3. Streamlit Port Already Used
   Port 8501 already in use
   Solusi:
   - Gunakan port lain: --server.port 8502
   - Atau kill process: pkill -f streamlit

4. Virtual Environment Issues
   Untuk fish shell:
   - Gunakan activate.fish bukan activate
   - Atau install virtualfish: pip install virtualfish

FITUR LANJUTAN
--------------

1. Adaptive UI
   - Form dengan clear_on_submit=True untuk auto-clear input
   - Enter key support (tidak perlu klik tombol)
   - Streaming effect untuk response

2. Error Resilience
   - Connection retry logic
   - Graceful degradation
   - User-friendly error messages

3. Extensibility
   - Mudah tambah model baru
   - Customizable system prompts
   - Modular code structure

PERFORMANCE NOTES
-----------------
- Timeout: 120 detik untuk long responses
- Connection pooling untuk efficient API calls
- Session state optimization untuk chat history
- Minimal dependencies untuk fast startup

KONTRIBUSI
----------
1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Buat Pull Request

LISENSI
-------
Proyek ini untuk tujuan edukasi dan testing.

KONTAK
------
Untuk pertanyaan atau issues, silakan buka issue di repository.

---
Last Updated: April 11, 2026
Version: 1.0.0
Status: Production Ready
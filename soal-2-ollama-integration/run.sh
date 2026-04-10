#!/bin/bash

# Script untuk menjalankan Ollama Chat Assistant

echo "🚀 Starting Ollama Chat Assistant..."

# Cek apakah virtual environment ada
if [ ! -d "venv_ollama" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python -m venv venv_ollama
fi

# Aktifkan virtual environment
source venv_ollama/bin/activate

# Install dependencies jika belum
if [ ! -f "venv_ollama/installed" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    touch venv_ollama/installed
fi

# Cek apakah Ollama berjalan
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama service not running."
    echo ""
    echo "PILIHAN:"
    echo "1. Jalankan Ollama dengan Docker (direkomendasikan)"
    echo "2. Install Ollama manual (butuh sudo)"
    echo "3. Lanjut tanpa Ollama (UI only)"
    echo ""
    read -p "Pilih opsi (1/2/3): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            echo "🐳 Starting Ollama via Docker..."
            if ! docker ps > /dev/null 2>&1; then
                echo "❌ Docker tidak berjalan. Starting Docker service..."
                sudo systemctl start docker 2>/dev/null || echo "⚠️  Gagal start Docker"
            fi
            
            # Cek apakah container Ollama sudah running
            if ! docker ps | grep -q ollama; then
                echo "📦 Pulling Ollama Docker image..."
                docker pull ollama/ollama:latest
                echo "🚀 Starting Ollama container..."
                docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:latest
                echo "⏳ Waiting for Ollama to start..."
                sleep 10
            else
                echo "✅ Ollama container sudah berjalan"
            fi
            ;;
        2)
            echo "📥 Installing Ollama manually..."
            echo "⚠️  Butuh sudo password"
            curl -fsSL https://ollama.ai/install.sh | sh
            echo "🚀 Starting Ollama service..."
            ollama serve &
            sleep 5
            ;;
        3)
            echo "⚠️  Running in UI-only mode (no Ollama)"
            echo "📝 Note: Chat functionality will be limited"
            ;;
        *)
            echo "❌ Pilihan tidak valid. Aborted."
            exit 1
            ;;
    esac
fi

# Cek kembali apakah Ollama berjalan
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service is running"
    echo "🔍 Checking available models..."
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; data=json.load(sys.stdin); print('Available models:', ', '.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null || echo "No models found")
    echo "$MODELS"
    
    # Jika tidak ada model, download default
    if echo "$MODELS" | grep -q "No models found" || echo "$MODELS" | grep -q "Available models: "; then
        if [[ $(echo "$MODELS" | wc -c) -lt 20 ]]; then
            echo "📥 Downloading default model (llama3.2)..."
            if command -v docker > /dev/null 2>&1 && docker ps | grep -q ollama; then
                docker exec ollama ollama pull llama3.2
            elif command -v ollama > /dev/null 2>&1; then
                ollama pull llama3.2
            else
                echo "⚠️  Cannot download model: Ollama not available"
            fi
        fi
    fi
else
    echo "⚠️  Ollama not available. Running in limited mode."
fi

echo ""

# Jalankan Streamlit
echo "🌐 Starting Streamlit on http://localhost:8501"
echo "📝 Press Ctrl+C to stop"
echo ""

streamlit run app.py --server.port 8501
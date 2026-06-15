#!/usr/bin/env bash
set -euo pipefail

INSTALL_TESSERACT=false
for arg in "$@"; do
    [[ "$arg" == "--install-tesseract" ]] && INSTALL_TESSERACT=true
done

echo ""
echo "Medical DeID Integrated - Setup"
echo "================================"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if ! command -v python3 &>/dev/null; then
    echo "[FAIL] Python no esta disponible en PATH."
    echo "Instala Python 3.10+ y volve a correr este script."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[INFO] Creando entorno virtual..."
    python3 -m venv .venv
else
    echo "[OK] Entorno virtual existente encontrado."
fi

PYTHON_EXE="$PROJECT_ROOT/.venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    echo "[FAIL] No se encontro .venv/bin/python"
    exit 1
fi

echo "[INFO] Actualizando pip..."
"$PYTHON_EXE" -m pip install --upgrade pip

echo "[INFO] Instalando dependencias del proyecto..."
"$PYTHON_EXE" -m pip install -r requirements.txt

if [ "$INSTALL_TESSERACT" = true ]; then
    if command -v brew &>/dev/null; then
        echo "[INFO] Instalando Tesseract via Homebrew..."
        brew install tesseract
    elif command -v apt-get &>/dev/null; then
        echo "[INFO] Instalando Tesseract via apt..."
        sudo apt-get install -y tesseract-ocr
    else
        echo "[WARN] No se encontro un gestor de paquetes compatible. Instala Tesseract manualmente."
    fi
fi

echo "[INFO] Corriendo chequeo de entorno..."
"$PYTHON_EXE" check_env.py

echo ""
echo "Proximos pasos:"
echo "1. Activar el entorno: source .venv/bin/activate"
echo "2. Si vas a reentrenar, copiar yolov8s.pt a models/yolov8s.pt"
echo "3. Entrenar: python -m app.cli train"
echo "4. Probar con samples/: python -m app.cli smoke"
echo "5. Levantar UI: python -m app.cli serve --host 127.0.0.1 --port 8000"

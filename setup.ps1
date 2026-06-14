param(
    [switch]$InstallTesseract
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Medical DeID Integrated - Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[FAIL] Python no esta disponible en PATH." -ForegroundColor Red
    Write-Host "Instala Python 3.10+ y volve a correr este script." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[OK] Entorno virtual existente encontrado." -ForegroundColor Green
}

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[FAIL] No se encontro .venv\Scripts\python.exe" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Actualizando pip..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip

Write-Host "[INFO] Instalando dependencias del proyecto..." -ForegroundColor Yellow
& $pythonExe -m pip install -r requirements.txt

if ($InstallTesseract) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "[INFO] Instalando Tesseract via winget..." -ForegroundColor Yellow
        winget install UB-Mannheim.TesseractOCR
    } else {
        Write-Host "[WARN] winget no esta disponible. Instala Tesseract manualmente." -ForegroundColor Yellow
    }
}

Write-Host "[INFO] Corriendo chequeo de entorno..." -ForegroundColor Yellow
& $pythonExe check_env.py

Write-Host ""
Write-Host "Proximos pasos:" -ForegroundColor Cyan
Write-Host "1. Activar el entorno: .\.venv\Scripts\Activate.ps1"
Write-Host "2. Si vas a reentrenar, copiar yolov8s.pt a models\yolov8s.pt"
Write-Host "3. Entrenar: python -m app.cli train"
Write-Host "4. Probar con samples/: python -m app.cli smoke"
Write-Host "5. Levantar UI: python -m app.cli serve --host 127.0.0.1 --port 8000"

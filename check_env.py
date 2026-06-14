from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FULL_DATASET_DIR = PROJECT_ROOT / "full_dataset" / "hackathon_TREE_AIBiomed"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_PYTHON = (3, 10)
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "cv2",
    "numpy",
    "pytesseract",
    "ultralytics",
    "torch",
]


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_python() -> bool:
    version = sys.version_info
    if version < REQUIRED_PYTHON:
        fail(
            f"Python {version.major}.{version.minor} detectado. "
            f"Se recomienda {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} o superior."
        )
        return False
    ok(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_packages() -> bool:
    success = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            ok(f"Paquete disponible: {package}")
        except Exception:
            fail(f"Falta instalar el paquete: {package}")
            success = False
    return success


def check_tesseract() -> bool:
    env_cmd = os.getenv("TESSERACT_CMD", "").strip()
    candidates = []
    if env_cmd:
        candidates.append(env_cmd)
    detected = shutil.which("tesseract")
    if detected:
        candidates.append(detected)
    candidates.extend(
        [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            ok(f"Tesseract encontrado: {candidate}")
            return True

    fail("Tesseract no esta instalado o no esta en PATH.")
    print("      Instalar sugerido en Windows:")
    print("      winget install UB-Mannheim.TesseractOCR")
    print("      Luego volver a correr este chequeo.")
    return False


def check_models() -> tuple[bool, bool]:
    base_ok = True
    best_present = False
    base_weights = MODELS_DIR / "yolov8s.pt"
    best_weights = MODELS_DIR / "best.pt"

    if base_weights.exists():
        ok(f"Peso base presente: {base_weights.name}")
    else:
        fail(
            f"Falta el peso base oficial de retraining: {base_weights}. "
            "Copialo en models/ o usa --base-weights."
        )
        legacy_nano = MODELS_DIR / "yolov8n.pt"
        if legacy_nano.exists():
            warn(
                f"Se encontro {legacy_nano.name}, pero la receta reproducible oficial "
                "ahora espera yolov8s.pt."
            )
        base_ok = False

    if best_weights.exists():
        ok(f"Modelo final presente: {best_weights.name}")
        best_present = True
    else:
        warn(
            "No se encontro models/best.pt. "
            "Todavia podes generarlo con 'python -m app.cli train'."
        )
    return base_ok, best_present


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def check_full_dataset() -> bool:
    success = True
    if FULL_DATASET_DIR.exists():
        ok(f"Full dataset presente: {FULL_DATASET_DIR.relative_to(PROJECT_ROOT)}")
    else:
        fail(
            "No se encontro full_dataset/hackathon_TREE_AIBiomed. "
            "Ese directorio es la unica fuente de verdad para training y validation."
        )
        return False

    data_yaml = FULL_DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        ok(f"Config de dataset presente: {data_yaml.relative_to(PROJECT_ROOT)}")
    else:
        fail(f"Falta data.yaml en {FULL_DATASET_DIR.relative_to(PROJECT_ROOT)}")
        success = False

    required_dirs = [
        FULL_DATASET_DIR / "images" / "train",
        FULL_DATASET_DIR / "images" / "val",
        FULL_DATASET_DIR / "labels" / "train",
        FULL_DATASET_DIR / "labels" / "val",
    ]
    for directory in required_dirs:
        if directory.exists():
            ok(f"Carpeta presente: {directory.relative_to(PROJECT_ROOT)}")
        else:
            fail(f"Falta carpeta: {directory.relative_to(PROJECT_ROOT)}")
            success = False

    train_images = count_files(FULL_DATASET_DIR / "images" / "train")
    val_images = count_files(FULL_DATASET_DIR / "images" / "val")
    train_labels = count_files(FULL_DATASET_DIR / "labels" / "train")
    val_labels = count_files(FULL_DATASET_DIR / "labels" / "val")

    print(f"Train images : {train_images}")
    print(f"Val images   : {val_images}")
    print(f"Train labels : {train_labels}")
    print(f"Val labels   : {val_labels}")

    if train_images == 0 or train_labels == 0:
        warn("El split de train parece incompleto o vacio.")
    if val_images == 0 or val_labels == 0:
        warn("El split de validacion parece incompleto o vacio.")

    return success


def check_samples() -> bool:
    samples_dir = PROJECT_ROOT / "samples"
    total = count_files(samples_dir)
    if total > 0:
        ok(f"Samples disponibles: {total}")
        return True
    warn("No hay samples cargados en samples/.")
    return True


def main() -> int:
    print("Medical DeID Integrated - Environment Check")
    print("=" * 42)

    checks = []

    print_section("Python")
    checks.append(check_python())

    print_section("Packages")
    checks.append(check_packages())

    print_section("Tesseract")
    checks.append(check_tesseract())

    print_section("Models")
    base_ok, best_present = check_models()
    checks.append(base_ok)

    print_section("Full Dataset")
    checks.append(check_full_dataset())

    print_section("Samples")
    checks.append(check_samples())

    print_section("Resultado")
    if all(checks):
        ok("El entorno esta listo para entrenar o probar el proyecto.")
        print("Siguiente paso sugerido:")
        if best_present:
            print("python -m app.cli smoke")
        else:
            print("python -m app.cli train")
        return 0

    fail("El entorno todavia no esta listo. Corregi los puntos marcados arriba y volve a correr este chequeo.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

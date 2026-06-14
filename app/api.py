from __future__ import annotations

import io
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import AppConfig
from app.core.pipeline import DeidentificationPipeline
from app.core.reporting import build_report


config = AppConfig.from_env()
config.ensure_runtime_dirs()
pipeline_default = DeidentificationPipeline(config)
pipeline_fallback = DeidentificationPipeline(
    replace(
        config,
        enable_full_image_fallback=True,
    )
)
app = FastAPI(
    title="Integrated Medical De-Identification API",
    version="1.0.0",
    description="YOLO detection + robust OCR + decision policy + redaction.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(config.static_dir)), name="static")


def _log(message: str) -> None:
    print(f"[API] {message}", flush=True)


def _format_model_summary() -> str:
    try:
        summary = pipeline_default.detector.describe_model()
    except Exception as exc:
        return f"Could not inspect detector metadata: {exc}"

    model_classes = summary["model_classes"]
    preview_items = list(model_classes.items())[:8]
    preview = ", ".join(f"{class_id}:{name}" for class_id, name in preview_items)
    if len(model_classes) > len(preview_items):
        preview += ", ..."
    return (
        f"weights={summary['model_path']}, aligned={summary['aligned_with_project_classes']}, "
        f"model_classes={preview}"
    )


@app.on_event("startup")
def log_startup_state() -> None:
    _log("Starting FastAPI server for Medical DeID Integrated.")
    _log(f"Project root: {config.project_root}")
    _log(f"Full dataset directory: {config.full_dataset_dir}")
    _log(f"Static UI directory: {config.static_dir}")
    _log(f"Outputs directory: {config.outputs_dir}")
    _log(
        f"Model files: best.pt exists={config.best_model_path.exists()} "
        f"at {config.best_model_path}; base exists={config.base_weights_path.exists()} "
        f"at {config.base_weights_path}"
    )
    _log(
        f"Tesseract command: {config.tesseract_cmd or 'auto-detect from PATH/common Windows installs'}"
    )
    _log(
        f"Redaction strategy: {config.redaction_strategy}, granularity={config.redaction_granularity}, "
        f"fallback={config.redaction_fallback}, compare={config.compare_inpainting_methods}, "
        f"padding(character={config.character_mask_padding}, token={config.token_mask_padding}, "
        f"roi={config.roi_mask_padding}), inpaint_radius={config.inpaint_radius}"
    )
    _log(f"Detector summary: {_format_model_summary()}")
    _log("UI available at '/'. API endpoints available at '/health', '/deidentify', and '/deidentify/report'.")


def _decode_upload(file: UploadFile, data: bytes) -> np.ndarray:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return image


def _select_pipeline(enable_fallback: bool) -> DeidentificationPipeline:
    return pipeline_fallback if enable_fallback else pipeline_default


@app.get("/health")
def health() -> dict:
    _log("Health check requested.")
    return {
        "status": "ok",
        "best_model_present": config.best_model_path.exists(),
        "base_weights_present": config.base_weights_path.exists(),
        "full_dataset_dir": str(config.full_dataset_dir),
    }


@app.post("/deidentify/report")
async def deidentify_report(
    file: UploadFile = File(...),
    enable_fallback: bool = Query(default=False),
) -> JSONResponse:
    started = time.time()
    contents = await file.read()
    _log(
        f"/deidentify/report request received: filename='{file.filename}', "
        f"bytes={len(contents)}, enable_fallback={enable_fallback}"
    )
    image = _decode_upload(file, contents)
    _log(f"Decoded image for report request with shape={image.shape}.")
    pipeline = _select_pipeline(enable_fallback)
    _log(f"Selected pipeline variant: {'fallback-enabled' if enable_fallback else 'default'}.")
    result = pipeline.run_on_image(image, file.filename or "upload")
    report = build_report(result, time.time() - started)
    _log(
        f"Report ready for '{report['filename']}': regions={report['summary']['regions_found']}, "
        f"redacted={report['summary']['redacted']}, review={report['summary']['review']}, "
        f"fallback_used={report['fallback_used']}"
    )
    return JSONResponse(report)


@app.post("/deidentify")
async def deidentify(
    file: UploadFile = File(...),
    enable_fallback: bool = Query(default=False),
) -> StreamingResponse:
    started = time.time()
    contents = await file.read()
    _log(
        f"/deidentify request received: filename='{file.filename}', "
        f"bytes={len(contents)}, enable_fallback={enable_fallback}"
    )
    image = _decode_upload(file, contents)
    _log(f"Decoded image for redaction request with shape={image.shape}.")
    pipeline = _select_pipeline(enable_fallback)
    _log(f"Selected pipeline variant: {'fallback-enabled' if enable_fallback else 'default'}.")
    result = pipeline.run_on_image(image, file.filename or "upload")
    report = build_report(result, time.time() - started)
    _log(
        f"Image response ready for '{report['filename']}': regions={report['summary']['regions_found']}, "
        f"redacted={report['summary']['redacted']}, review={report['summary']['review']}, "
        f"fallback_used={report['fallback_used']}"
    )

    ok, buffer = cv2.imencode(".png", result.redacted_image)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode redacted image.")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={
            "X-Report": json.dumps(report),
            "X-Processing-Time": str(report["processing_time_s"]),
            "X-Redacted-Count": str(report["summary"]["redacted"]),
            "X-Review-Count": str(report["summary"]["review"]),
        },
    )


@app.get("/")
def root() -> FileResponse:
    _log(f"Serving UI document from {Path(config.static_dir / 'index.html')}.")
    return FileResponse(config.static_dir / "index.html")

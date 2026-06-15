# file: app/cli.py
# description: Provides CLI commands for training, smoke testing, batch processing, visualization, and serving the app.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import numpy as np
import uvicorn

from app.core.config import AppConfig
from app.core.pipeline import DeidentificationPipeline
from app.core.reporting import build_report
from app.train import default_resume_checkpoint, train_model, validate_trained_model
from app.visualizer import visualize_split


def _iter_images(input_path: Path) -> list[Path]:
    """
    Resolves the list of input images from a file or directory path.

    Args:
        input_path (Path): Image file or directory containing images to process.

    Returns:
        list[Path]: Sorted image paths when a directory is provided, or a single-item list otherwise.
    """
    if input_path.is_dir():
        files: list[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            files.extend(sorted(input_path.glob(pattern)))
        return files
    return [input_path]


def cmd_train(args: argparse.Namespace) -> int:
    """
    Trains the YOLO detector and optionally validates the resulting model.

    Args:
        args (argparse.Namespace): Parsed CLI arguments for the training command.

    Returns:
        int: Process exit code, `0` on success.
    """
    best_path = train_model(
        dataset_root=args.dataset_root,
        base_weights_path=args.base_weights,
        profile=args.profile,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        resume=args.resume,
        resume_from=args.resume_from,
    )
    print(f"Training complete. best.pt saved to: {best_path}")
    if args.skip_validate:
        return 0
    metrics = validate_trained_model(dataset_root=args.dataset_root, img_size=args.img_size)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """
    Visualizes labeled dataset samples for manual inspection.

    Args:
        args (argparse.Namespace): Parsed CLI arguments for the visualization command.

    Returns:
        int: Process exit code, `0` on success.
    """
    saved = visualize_split(args.split, args.n, save=args.save, seed=args.seed)
    if args.save:
        for path in saved:
            print(path)
    return 0


def _build_runtime_config(
    base_config: AppConfig,
    *,
    enable_fallback: bool,
    save_debug_report: bool,
    character_mask_padding: int,
    token_mask_padding: int,
    roi_mask_padding: int,
) -> AppConfig:
    """
    Clones the base configuration with runtime overrides used by CLI processing commands.

    Args:
        base_config (AppConfig): Base configuration loaded from the environment.
        enable_fallback (bool): Whether full-image fallback OCR should be enabled.
        save_debug_report (bool): Whether OCR debug details should be persisted in debug JSON.
        character_mask_padding (int): Character-mask padding override in pixels.
        token_mask_padding (int): Token-mask padding override in pixels.
        roi_mask_padding (int): ROI fallback padding override in pixels.

    Returns:
        AppConfig: Updated configuration instance with CLI-specific overrides applied.
    """
    return replace(
        base_config,
        enable_full_image_fallback=enable_fallback,
        save_debug_report=save_debug_report,
        character_mask_padding=character_mask_padding,
        token_mask_padding=token_mask_padding,
        roi_mask_padding=roi_mask_padding,
    )


def _mask_to_image(mask: np.ndarray | None) -> np.ndarray | None:
    """
    Normalizes an internal mask into an 8-bit image for writing to disk.

    Args:
        mask (np.ndarray | None): Mask returned by the redactor, or `None`.

    Returns:
        np.ndarray | None: 8-bit mask image, or `None` when no mask is available.
    """
    if mask is None or mask.size == 0:
        return None
    return (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask


def _build_mask_overlay(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    """
    Builds a semi-transparent red overlay highlighting masked pixels.

    Args:
        image (np.ndarray): Original BGR image.
        mask (np.ndarray | None): Binary redaction mask, or `None`.

    Returns:
        np.ndarray | None: Overlay visualization, or `None` when no mask is available.
    """
    mask_image = _mask_to_image(mask)
    if mask_image is None:
        return None
    overlay = image.copy()
    overlay[mask_image > 0] = (0, 0, 255)
    return cv2.addWeighted(image, 0.72, overlay, 0.28, 0.0)


def _write_redaction_artifacts(
    *,
    output_path: Path,
    image_path: Path,
    result_image: np.ndarray,
    original_image: np.ndarray,
    mask: np.ndarray | None,
    save_redaction_debug: bool,
) -> Path:
    """
    Writes the redacted image and optional mask/overlay debug artifacts.

    Args:
        output_path (Path): Directory where artifacts must be written.
        image_path (Path): Original image path used to derive output names.
        result_image (np.ndarray): Final redacted image.
        original_image (np.ndarray): Original image used for overlay rendering.
        mask (np.ndarray | None): Combined redaction mask, or `None`.
        save_redaction_debug (bool): Whether to persist mask and overlay artifacts.

    Returns:
        Path: Path to the main redacted output image written to disk.
    """
    primary_output = output_path / f"redacted_{image_path.name}"
    cv2.imwrite(str(primary_output), result_image)

    if save_redaction_debug:
        mask_image = _mask_to_image(mask)
        if mask_image is not None:
            cv2.imwrite(str(output_path / f"mask_{image_path.stem}.png"), mask_image)
            overlay = _build_mask_overlay(original_image, mask)
            if overlay is not None:
                cv2.imwrite(str(output_path / f"overlay_{image_path.stem}.png"), overlay)

    return primary_output


def _write_debug_payload(
    *,
    output_path: Path,
    image_path: Path,
    report: dict,
    result,
    save_debug_report: bool,
) -> None:
    """
    Writes a JSON debug payload with detection- and OCR-level diagnostics.

    Args:
        output_path (Path): Directory where the debug JSON will be written.
        image_path (Path): Original image path used to derive the debug filename.
        report (dict): Serialized report payload for the processed image.
        result: Pipeline result object containing OCR and redaction details.
        save_debug_report (bool): Whether OCR pass-level debug data should be included.

    Returns:
        None: The debug payload is persisted to disk.
    """
    debug_path = output_path / f"debug_{image_path.stem}.json"
    debug_payload = {
        "filename": result.filename,
        "redaction_methods_run": report["redaction_methods_run"],
        "redaction_mask_pixel_count": report["redaction_mask_pixel_count"],
        "detections": [
            {
                "class_name": item.detection.class_name,
                "decision": item.decision.action,
                "redaction_method": item.redaction_method,
                "mask_source": item.mask_source,
                "mask_pixel_count": item.mask_pixel_count,
                "ocr_character_count": len(item.ocr.characters),
                "ocr_token_count": len(item.ocr.tokens),
                "ocr_debug": item.ocr.debug if save_debug_report else None,
            }
            for item in result.processed_detections
        ],
        "fallback_ocr_debug": result.fallback_ocr.debug if save_debug_report and result.fallback_ocr else None,
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")


def _run_pipeline_for_images(
    *,
    input_path: Path,
    output_path: Path,
    enable_fallback: bool,
    save_debug_report: bool,
    save_redaction_debug: bool,
    character_mask_padding: int,
    token_mask_padding: int,
    roi_mask_padding: int,
) -> int:
    """
    Runs the pipeline over one or more images and writes outputs to disk.

    Args:
        input_path (Path): Image file or directory to process.
        output_path (Path): Directory where images and reports will be written.
        enable_fallback (bool): Whether full-image fallback OCR should be enabled.
        save_debug_report (bool): Whether OCR debug data should be included in debug JSON.
        save_redaction_debug (bool): Whether mask and overlay images should be written.
        character_mask_padding (int): Character-mask padding override in pixels.
        token_mask_padding (int): Token-mask padding override in pixels.
        roi_mask_padding (int): ROI fallback padding override in pixels.

    Returns:
        int: Process exit code, `0` on success.
    """
    base_config = AppConfig.from_env()
    config = _build_runtime_config(
        base_config,
        enable_fallback=enable_fallback,
        save_debug_report=save_debug_report,
        character_mask_padding=character_mask_padding,
        token_mask_padding=token_mask_padding,
        roi_mask_padding=roi_mask_padding,
    )
    pipeline = DeidentificationPipeline(config)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in _iter_images(input_path):
        started = time.time()
        result = pipeline.run_on_path(image_path)
        report = build_report(result, time.time() - started)
        out_report = output_path / f"report_{image_path.stem}.json"
        original_image = cv2.imread(str(image_path))
        out_image = _write_redaction_artifacts(
            output_path=output_path,
            image_path=image_path,
            result_image=result.redacted_image,
            original_image=original_image if original_image is not None else result.redacted_image,
            mask=result.redaction_mask,
            save_redaction_debug=save_redaction_debug,
        )
        out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if save_debug_report or save_redaction_debug:
            _write_debug_payload(
                output_path=output_path,
                image_path=image_path,
                report=report,
                result=result,
                save_debug_report=save_debug_report,
            )
        print(f"[OK] {image_path.name} -> {out_image.name}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """
    Processes a user-provided image or directory through the full pipeline.

    Args:
        args (argparse.Namespace): Parsed CLI arguments for the `run` command.

    Returns:
        int: Process exit code, `0` on success.
    """
    return _run_pipeline_for_images(
        input_path=args.input,
        output_path=args.output,
        enable_fallback=args.enable_full_image_fallback,
        save_debug_report=args.save_debug_report,
        save_redaction_debug=args.save_redaction_debug,
        character_mask_padding=args.character_mask_padding,
        token_mask_padding=args.token_mask_padding,
        roi_mask_padding=args.roi_mask_padding,
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    """
    Runs a smoke test over `samples/` or a user-provided input image.

    Args:
        args (argparse.Namespace): Parsed CLI arguments for the `smoke` command.

    Returns:
        int: Process exit code, `0` on success.
    """
    config = AppConfig.from_env()
    output = config.outputs_dir / "smoke"
    input_path = args.input or (config.project_root / "samples")
    return _run_pipeline_for_images(
        input_path=input_path,
        output_path=output,
        enable_fallback=True,
        save_debug_report=args.save_debug_report,
        save_redaction_debug=args.save_redaction_debug,
        character_mask_padding=args.character_mask_padding,
        token_mask_padding=args.token_mask_padding,
        roi_mask_padding=args.roi_mask_padding,
    )


def cmd_serve(args: argparse.Namespace) -> int:
    """
    Starts the FastAPI application and static UI server.

    Args:
        args (argparse.Namespace): Parsed CLI arguments for the `serve` command.

    Returns:
        int: Process exit code, `0` when the server stops cleanly.
    """
    config = AppConfig.from_env()
    print("[Serve] Starting FastAPI + UI server.", flush=True)
    print(f"[Serve] UI URL: http://{args.host}:{args.port}/", flush=True)
    print(f"[Serve] API health endpoint: http://{args.host}:{args.port}/health", flush=True)
    print(f"[Serve] Expected detector weights: {config.best_model_path}", flush=True)
    print(
        "[Serve] If you see classes outside name/id/age/date/time in the logs, "
        "the loaded best.pt does not match this medical dataset.",
        flush=True,
    )
    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the top-level argument parser for all CLI commands.

    Args:
        None

    Returns:
        argparse.ArgumentParser: Configured parser with all subcommands and arguments.
    """
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Integrated medical image de-identification CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a YOLO detector using the local full_dataset.")
    train_parser.add_argument("--dataset-root", type=Path, default=config.full_dataset_dir)
    train_parser.add_argument("--base-weights", type=Path, default=config.base_weights_path)
    train_parser.add_argument("--profile", default="historical")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--img-size", type=int, default=640)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help=f"Resume from a specific checkpoint. Default resume checkpoint is {default_resume_checkpoint(config)}",
    )
    train_parser.add_argument("--skip-validate", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    visualize_parser = subparsers.add_parser("visualize", help="Visualize labels from full_dataset.")
    visualize_parser.add_argument("--split", choices=["train", "val"], default="train")
    visualize_parser.add_argument("--n", type=int, default=5)
    visualize_parser.add_argument("--save", action="store_true")
    visualize_parser.add_argument("--seed", type=int, default=42)
    visualize_parser.set_defaults(func=cmd_visualize)

    run_parser = subparsers.add_parser("run", help="Run the integrated pipeline on a file or folder.")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--enable-full-image-fallback", action="store_true")
    run_parser.add_argument("--save-debug-report", action="store_true")
    run_parser.add_argument("--save-redaction-debug", action="store_true")
    run_parser.add_argument("--character-mask-padding", type=int, default=config.character_mask_padding)
    run_parser.add_argument("--token-mask-padding", type=int, default=config.token_mask_padding)
    run_parser.add_argument("--roi-mask-padding", type=int, default=config.roi_mask_padding)
    run_parser.set_defaults(func=cmd_run)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run a smoke test over samples/ by default, or a custom image path.",
    )
    smoke_parser.add_argument("--input", type=Path)
    smoke_parser.add_argument("--save-debug-report", action="store_true")
    smoke_parser.add_argument("--save-redaction-debug", action="store_true")
    smoke_parser.add_argument("--character-mask-padding", type=int, default=config.character_mask_padding)
    smoke_parser.add_argument("--token-mask-padding", type=int, default=config.token_mask_padding)
    smoke_parser.add_argument("--roi-mask-padding", type=int, default=config.roi_mask_padding)
    smoke_parser.set_defaults(func=cmd_smoke)

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main() -> int:
    """
    Parses CLI arguments and dispatches execution to the selected command handler.

    Args:
        None

    Returns:
        int: Process exit code returned by the selected command.
    """
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

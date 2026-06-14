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
    if input_path.is_dir():
        files: list[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            files.extend(sorted(input_path.glob(pattern)))
        return files
    return [input_path]


def cmd_train(args: argparse.Namespace) -> int:
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
    redaction_strategy: str,
    compare_inpainting_methods: bool,
    character_mask_padding: int,
    token_mask_padding: int,
    roi_mask_padding: int,
) -> AppConfig:
    return replace(
        base_config,
        enable_full_image_fallback=enable_fallback,
        save_debug_report=save_debug_report,
        redaction_strategy=redaction_strategy,
        compare_inpainting_methods=compare_inpainting_methods,
        character_mask_padding=character_mask_padding,
        token_mask_padding=token_mask_padding,
        roi_mask_padding=roi_mask_padding,
    )


def _mask_to_image(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None or mask.size == 0:
        return None
    return (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask


def _build_mask_overlay(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    mask_image = _mask_to_image(mask)
    if mask_image is None:
        return None
    overlay = image.copy()
    overlay[mask_image > 0] = (0, 0, 255)
    return cv2.addWeighted(image, 0.72, overlay, 0.28, 0.0)


def _annotate_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (max(160, len(label) * 11), 36), (255, 255, 255), -1)
    cv2.putText(panel, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return panel


def _build_comparison_sheet(
    original: np.ndarray,
    mask: np.ndarray | None,
    variants: dict[str, np.ndarray],
) -> np.ndarray | None:
    overlay = _build_mask_overlay(original, mask)
    if overlay is None or not variants:
        return None

    panels = [
        _annotate_panel(original, "original"),
        _annotate_panel(cv2.cvtColor(_mask_to_image(mask), cv2.COLOR_GRAY2BGR), "mask"),
        _annotate_panel(overlay, "overlay"),
    ]
    panels.extend(_annotate_panel(image, method) for method, image in variants.items())

    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    normalized: list[np.ndarray] = []
    for panel in panels:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        normalized.append(canvas)

    if len(normalized) % 2 != 0:
        normalized.append(np.full((height, width, 3), 255, dtype=np.uint8))

    rows = []
    for index in range(0, len(normalized), 2):
        rows.append(np.hstack(normalized[index : index + 2]))
    return np.vstack(rows)


def _write_redaction_artifacts(
    *,
    output_path: Path,
    image_path: Path,
    result_image: np.ndarray,
    original_image: np.ndarray,
    variants: dict[str, np.ndarray],
    methods_run: list[str],
    preferred_method: str,
    mask: np.ndarray | None,
    compare_mode: bool,
    save_redaction_debug: bool,
) -> Path:
    if compare_mode and variants:
        primary_method = preferred_method if preferred_method in variants else methods_run[0]
        primary_output = output_path / f"redacted_{image_path.stem}_{primary_method}.png"
        for method, image in variants.items():
            variant_path = output_path / f"redacted_{image_path.stem}_{method}.png"
            cv2.imwrite(str(variant_path), image)
    else:
        primary_output = output_path / f"redacted_{image_path.name}"
        cv2.imwrite(str(primary_output), result_image)

    if save_redaction_debug:
        mask_image = _mask_to_image(mask)
        if mask_image is not None:
            cv2.imwrite(str(output_path / f"mask_{image_path.stem}.png"), mask_image)
            overlay = _build_mask_overlay(original_image, mask)
            if overlay is not None:
                cv2.imwrite(str(output_path / f"overlay_{image_path.stem}.png"), overlay)
        comparison = _build_comparison_sheet(original_image, mask, variants)
        if comparison is not None:
            cv2.imwrite(str(output_path / f"comparison_{image_path.stem}.png"), comparison)

    return primary_output


def _write_debug_payload(
    *,
    output_path: Path,
    image_path: Path,
    report: dict,
    result,
    save_debug_report: bool,
) -> None:
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
    redaction_strategy: str,
    compare_inpainting_methods: bool,
    character_mask_padding: int,
    token_mask_padding: int,
    roi_mask_padding: int,
) -> int:
    base_config = AppConfig.from_env()
    config = _build_runtime_config(
        base_config,
        enable_fallback=enable_fallback,
        save_debug_report=save_debug_report,
        redaction_strategy=redaction_strategy,
        compare_inpainting_methods=compare_inpainting_methods,
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
            variants=result.redacted_variants,
            methods_run=result.redaction_methods_run,
            preferred_method=config.redaction_strategy,
            mask=result.redaction_mask,
            compare_mode=config.compare_inpainting_methods,
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
    return _run_pipeline_for_images(
        input_path=args.input,
        output_path=args.output,
        enable_fallback=args.enable_full_image_fallback,
        save_debug_report=args.save_debug_report,
        save_redaction_debug=args.save_redaction_debug,
        redaction_strategy=args.redaction_strategy,
        compare_inpainting_methods=args.compare_inpainting_methods,
        character_mask_padding=args.character_mask_padding,
        token_mask_padding=args.token_mask_padding,
        roi_mask_padding=args.roi_mask_padding,
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    config = AppConfig.from_env()
    output = config.outputs_dir / "smoke"
    input_path = args.input or (config.project_root / "samples")
    return _run_pipeline_for_images(
        input_path=input_path,
        output_path=output,
        enable_fallback=True,
        save_debug_report=args.save_debug_report,
        save_redaction_debug=args.save_redaction_debug,
        redaction_strategy=args.redaction_strategy,
        compare_inpainting_methods=args.compare_inpainting_methods,
        character_mask_padding=args.character_mask_padding,
        token_mask_padding=args.token_mask_padding,
        roi_mask_padding=args.roi_mask_padding,
    )


def cmd_serve(args: argparse.Namespace) -> int:
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
    run_parser.add_argument("--redaction-strategy", choices=["biharmonic", "telea", "ns"], default=config.redaction_strategy)
    run_parser.add_argument("--compare-inpainting-methods", action="store_true")
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
    smoke_parser.add_argument(
        "--redaction-strategy",
        choices=["biharmonic", "telea", "ns"],
        default=config.redaction_strategy,
    )
    smoke_parser.add_argument("--compare-inpainting-methods", action="store_true")
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
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

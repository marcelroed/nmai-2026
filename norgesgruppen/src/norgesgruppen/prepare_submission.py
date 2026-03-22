"""Prepare a submission zip from a trained RF-DETR 2XLarge checkpoint.

Steps:
1. Load the trained checkpoint (EMA weights) into a fresh MODEL_CLS model
2. Export to ONNX
3. Convert ONNX to FP16 (to fit within 420 MB weight limit)
4. Package run.py + FP16 ONNX model into a zip

Usage:
    uv run prepare-submission --checkpoint checkpoints/checkpoint_xxlarge3_ep19.pth
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import onnx
import torch
import typer
from onnxruntime.transformers.float16 import convert_float_to_float16

from norgesgruppen.config import MODEL_CLS, NUM_CLASSES, RESOLUTION

SUBMISSION_DIR = Path("submission")


BATCH_SIZE = 16


def main(
    checkpoint: Path = Path("checkpoints/checkpoint_xxlarge3_ep19.pth"),
    output_zip: Path = Path("submission.zip"),
):
    print(f"Loading checkpoint: {checkpoint}")
    model = MODEL_CLS()
    # rfdetr convention: num_classes = actual_categories + 1 (background)
    model.model.reinitialize_detection_head(num_classes=NUM_CLASSES + 1)

    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Prefer EMA weights (better generalization)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.model.model.load_state_dict(ckpt[state_key])
    print(f"Loaded {state_key} weights (epoch {ckpt.get('epoch', '?')})")

    # Export to ONNX (FP32)
    onnx_dir = SUBMISSION_DIR / "onnx_export"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.export(output_dir=str(onnx_dir), batch_size=BATCH_SIZE)
    print(f"Exported ONNX to {onnx_dir}")

    # Find the exported ONNX file (inference_model.onnx, not any pre-existing fp16)
    onnx_path = onnx_dir / "inference_model.onnx"
    if not onnx_path.exists():
        onnx_files = [f for f in onnx_dir.glob("*.onnx") if "fp16" not in f.name]
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX file found in {onnx_dir}")
        onnx_path = onnx_files[0]
    fp32_size = onnx_path.stat().st_size / 1e6
    print(f"FP32 ONNX: {onnx_path} ({fp32_size:.1f} MB)")

    # Convert to FP16 (keep I/O types as FP32 so preprocessing stays unchanged)
    print("Converting to FP16...")
    onnx_model = onnx.load(str(onnx_path))
    # Use ORT's FP16 converter (onnxconverter_common's version creates type mismatches in Cast/Add nodes).
    # decode_single in run.py casts outputs to float32 before numpy math.
    onnx_model_fp16 = convert_float_to_float16(onnx_model, keep_io_types=False)

    fp16_path = onnx_path.with_name("model_fp16.onnx")
    onnx.save(onnx_model_fp16, str(fp16_path))
    fp16_size = fp16_path.stat().st_size / 1e6
    print(f"FP16 ONNX: {fp16_path} ({fp16_size:.1f} MB, {fp32_size - fp16_size:.1f} MB saved)")

    if fp16_size > 420:
        print(f"WARNING: FP16 model is {fp16_size:.1f} MB, exceeds 420 MB limit!")

    # Package submission
    submission_root = SUBMISSION_DIR / "package"
    submission_root.mkdir(parents=True, exist_ok=True)

    # Copy run.py
    run_py_src = Path(__file__).parent.parent.parent / "submission" / "run.py"
    if not run_py_src.exists():
        run_py_src = Path("submission/run.py")
    shutil.copy2(run_py_src, submission_root / "run.py")

    # Copy FP16 ONNX model
    shutil.copy2(fp16_path, submission_root / "model.onnx")

    # Create zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in submission_root.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(submission_root)
                zf.write(file, arcname)

    zip_size = output_zip.stat().st_size / 1e6
    print(f"\nSubmission zip: {output_zip} ({zip_size:.1f} MB)")
    if zip_size > 420:
        print(f"WARNING: zip is {zip_size:.1f} MB, exceeds 420 MB limit!")
    else:
        print(f"OK: {420 - zip_size:.1f} MB under the limit")


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()

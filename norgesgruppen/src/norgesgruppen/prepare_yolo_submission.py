"""Prepare a submission zip from a trained YOLOv26 checkpoint.

Steps:
1. Load the trained YOLO checkpoint
2. Export to ONNX (FP16, end2end for NMS-free inference)
3. Package run.py + ONNX model + class_mapping.json into a zip

Usage:
    uv run prepare-yolo-submission --checkpoint output-yolo26x/yolo_run/weights/best.pt
    uv run prepare-yolo-submission --checkpoint output-yolo26x/best_competition.pt
"""

import json
import shutil
import zipfile
from pathlib import Path

import typer


def main(
    checkpoint: Path = Path("output-yolo26x/yolo_run/weights/best.pt"),
    output_zip: Path = Path("submission_yolo.zip"),
    imgsz: int = 640,
    class_mapping: str = "",
    half: bool = True,
    opset: int = 17,
):
    """Export YOLO checkpoint to ONNX and package for submission.

    Args:
        checkpoint: Path to YOLO .pt checkpoint.
        output_zip: Output zip path.
        imgsz: Image size for ONNX export (must match training).
        class_mapping: Path to class_mapping.json. If empty, auto-detected
            from the output directory.
        half: Export as FP16 (recommended for L4 GPU).
        opset: ONNX opset version (max 20 for sandbox).
    """
    from ultralytics import YOLO

    # Round imgsz to multiple of 32
    stride = 32
    if imgsz % stride != 0:
        imgsz = ((imgsz + stride - 1) // stride) * stride
        print(f"  Rounded imgsz to {imgsz} (must be multiple of {stride})")

    print(f"Loading checkpoint: {checkpoint}")
    model = YOLO(str(checkpoint))

    # Export to ONNX
    print(f"Exporting to ONNX (imgsz={imgsz}, half={half}, opset={opset})...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        opset=opset,
        simplify=True,
    )
    onnx_path = Path(onnx_path)
    onnx_size = onnx_path.stat().st_size / 1e6
    print(f"ONNX model: {onnx_path} ({onnx_size:.1f} MB)")

    if onnx_size > 420:
        print(f"WARNING: ONNX model is {onnx_size:.1f} MB, exceeds 420 MB limit!")

    # Find class mapping
    if not class_mapping:
        # Try to auto-detect from the training output directory
        candidates = [
            checkpoint.parent / "class_mapping.json",
            checkpoint.parent.parent / "class_mapping.json",
            checkpoint.parent.parent / "yolo_dataset" / "class_mapping.json",
            checkpoint.parent.parent.parent / "yolo_dataset" / "class_mapping.json",
        ]
        for c in candidates:
            if c.exists():
                class_mapping = str(c)
                break

    if not class_mapping or not Path(class_mapping).exists():
        print("WARNING: No class_mapping.json found! Submission will use raw YOLO class IDs.")
        print("  Provide --class-mapping path/to/class_mapping.json")
        mapping_data = None
    else:
        print(f"Class mapping: {class_mapping}")
        with open(class_mapping) as f:
            mapping_data = json.load(f)

    # Package submission
    submission_root = Path("submission_yolo_package")
    submission_root.mkdir(parents=True, exist_ok=True)

    # Copy run.py
    run_py_src = Path(__file__).parent.parent.parent / "submission_yolo" / "run.py"
    if not run_py_src.exists():
        run_py_src = Path("submission_yolo/run.py")
    if not run_py_src.exists():
        raise FileNotFoundError(
            f"Cannot find submission_yolo/run.py at {run_py_src}. "
            "Create it first."
        )
    shutil.copy2(run_py_src, submission_root / "run.py")

    # Copy ONNX model
    shutil.copy2(onnx_path, submission_root / "model.onnx")

    # Write class mapping
    if mapping_data:
        with open(submission_root / "class_mapping.json", "w") as f:
            json.dump(mapping_data, f)

    # Write config
    config = {"imgsz": imgsz, "half": half}
    with open(submission_root / "config.json", "w") as f:
        json.dump(config, f)

    # Create zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in submission_root.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(submission_root)
                zf.write(file, arcname)

    zip_size = output_zip.stat().st_size / 1e6
    print(f"\nSubmission zip: {output_zip} ({zip_size:.1f} MB)")

    # List contents
    with zipfile.ZipFile(output_zip) as zf:
        print("Contents:")
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size / 1e6:.1f} MB)")

    if zip_size > 420:
        print(f"WARNING: zip is {zip_size:.1f} MB, exceeds 420 MB limit!")
    else:
        print(f"OK: {420 - zip_size:.1f} MB under the limit")

    # Cleanup
    shutil.rmtree(submission_root)


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()

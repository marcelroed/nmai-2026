"""Prepares an RF-DETR model checkpoint for use by the evaluator.

Exports as a torch.export ExportedProgram in bfloat16 for fast inference.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import typer
from rfdetr import RFDETR2XLarge, RFDETRLarge

if TYPE_CHECKING:
    Model = RFDETRLarge
else:
    Model = RFDETR2XLarge


class InferenceWrapper(torch.nn.Module):
    """Thin wrapper that takes a plain [B,3,H,W] tensor and returns (boxes, logits)."""

    def __init__(self, lwdetr: torch.nn.Module):
        super().__init__()
        self.model = lwdetr

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["pred_boxes"], out["pred_logits"]


def main(
    pth_path: Path,
    output_path: Path = Path() / "submission" / "model.pt2",
):
    model = Model()
    model.model.reinitialize_detection_head(num_classes=358)
    print("Trying to load model")

    torch.serialization.add_safe_globals([argparse.Namespace])

    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    print(state_dict.keys())
    for key in state_dict["model"]:
        if state_dict["model"][key].is_floating_point():
            state_dict["model"][key] = state_dict["model"][key].to(torch.bfloat16)
    model.model.model.load_state_dict(state_dict["model"])
    print("Loaded model in bfloat16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lwdetr = model.model.model
    wrapper = InferenceWrapper(lwdetr)
    wrapper.to(device=device, dtype=torch.float32)
    wrapper.eval()

    # Install compile-safe patches to eliminate data-dependent ops
    from norgesgruppen.training import install_compile_safe_deform_patches
    spatial_side = 880 // 20  # resolution // patch_size for xxlarge
    install_compile_safe_deform_patches(spatial_side, spatial_side)

    # Export with torch.export (static batch=4, run.py pads to this).
    print(f"Exporting model with torch.export on {device}...")
    BATCH_SIZE = 4
    dummy = torch.randn(BATCH_SIZE, 3, 880, 880, device=device, dtype=torch.float32)
    with torch.no_grad():
        exported = torch.export.export(wrapper, (dummy,))

    # Verify
    print("Verifying...")
    with torch.no_grad():
        boxes, logits = exported.module()(dummy)
    print(f"  boxes: {boxes.shape}, logits: {logits.shape}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved exported model to {output_path} ({size_mb:.1f} MB)")


def _main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)

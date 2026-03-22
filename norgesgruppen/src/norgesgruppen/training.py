# Custom training loop, based on the code from rfdetr.

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import shutil
import socket
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Callable, DefaultDict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

# --- Performance: enable TF32 and cuDNN autotuning for fixed-size inputs ---
torch.set_float32_matmul_precision("high")  # TF32 on H200/Ampere+
torch.backends.cudnn.benchmark = True


class InterleavedDataset(Dataset):
    """Samples from two datasets with a configurable mixing ratio.

    Each __getitem__ flips a coin: with probability `synthetic_weight` it
    draws from the synthetic dataset, otherwise from the real dataset.
    Both datasets must return the same format (tensor, target_dict).
    """

    def __init__(self, dataset_real, dataset_synthetic, synthetic_weight=0.5,
                 samples_per_epoch=5000):
        self.real = dataset_real
        self.synthetic = dataset_synthetic
        self.synthetic_weight = synthetic_weight
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if random.random() < self.synthetic_weight:
            return self.synthetic[idx % len(self.synthetic)]
        return self.real[idx % len(self.real)]

try:
    from torch.amp import GradScaler, autocast

    _DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    _DEPRECATED_AMP = True

# ---------------------------------------------------------------------------
# Monkey-patch: fix torch.compile incompatibility in DINOv2 backbone
# ---------------------------------------------------------------------------
# F.interpolate(mode="bicubic", antialias=True) crashes the inductor backend
# when recompiling for a new output size (PyTorch bug in UpsampleBicubic2DAaBackward0).
# Dropping antialias=True is safe here: this interpolates a small positional
# encoding grid (e.g. 44×44 → 54×54), always upsampling, where anti-aliasing
# has no effect.
import rfdetr.models.backbone.dinov2_with_windowed_attn as _dinov2_mod
import rfdetr.util.misc as utils
from rfdetr.assets.model_weights import (
    download_pretrain_weights,
    validate_pretrain_weights,
)
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.detr import RFDETR
from rfdetr.engine import evaluate
from rfdetr.models import PostProcess, build_criterion_and_postprocessors, build_model
from rfdetr.models.lwdetr import LWDETR
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import NestedTensor, is_main_process, save_on_master
from rfdetr.util.utils import BestMetricHolder, ModelEma

from norgesgruppen import memeff_hungarian_matcher  # noqa: F401
from norgesgruppen.augmentation import AUG_CONFIG
from norgesgruppen.splitting import EXCLUDED_IMAGE_IDS, LABEL_MERGE_MAP

# ---------------------------------------------------------------------------
# Per-class loss reweighting
# ---------------------------------------------------------------------------

def compute_class_weights(dataset_dir: str, num_classes: int) -> torch.Tensor:
    """Compute per-class loss weights from training annotation frequencies.

    Uses sqrt inverse frequency: weight[c] = sqrt(median_freq / freq[c]),
    capped at [0.5, 3.0]. Classes with zero annotations get weight 1.0.
    Returns a tensor of shape [num_classes].
    """
    import json as _json
    from collections import Counter

    ann_path = Path(dataset_dir) / "train" / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = _json.load(f)

    counts = Counter(a["category_id"] for a in coco["annotations"])
    freqs = np.array([counts.get(c, 0) for c in range(num_classes)], dtype=np.float64)

    # Use median of non-zero frequencies as reference
    nonzero = freqs[freqs > 0]
    if len(nonzero) == 0:
        return torch.ones(num_classes)
    median_freq = float(np.median(nonzero))

    weights = np.ones(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if freqs[c] > 0:
            weights[c] = np.sqrt(median_freq / freqs[c])

    # Clamp to avoid extreme values
    weights = np.clip(weights, 0.5, 3.0)

    return torch.tensor(weights, dtype=torch.float32)


def _install_class_weighted_loss(criterion, class_weights: torch.Tensor):
    """Monkey-patch SetCriterion.loss_labels to apply per-class weights.

    Wraps the original loss_labels method. After the per-element loss tensor
    is computed (shape [B, Q, C]) and before it's summed, multiplies by
    the class weight vector.
    """
    original_loss_labels = criterion.loss_labels.__func__

    # Store weights on criterion so they move with .to(device) / .cuda()
    criterion.register_buffer("_class_weights", class_weights)

    def weighted_loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        # Call the original — it returns {"loss_ce": scalar, ...}
        # We need to intercept BEFORE the .sum(), so we replicate the ia_bce path
        # with the weight injection.
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            from rfdetr.util import box_ops

            alpha = self.focal_alpha
            gamma = 2
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob**gamma

            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)

            t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
            neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)

            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            # shape: [B, Q, C] — apply per-class weights before summing
            cw = self._class_weights.to(loss_ce.device)
            loss_ce = loss_ce * cw.unsqueeze(0).unsqueeze(0)
            loss_ce = loss_ce.sum() / num_boxes

            losses = {"loss_ce": loss_ce}
            if log:
                from rfdetr.util.misc import accuracy
                losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            return losses
        else:
            # For non-ia_bce paths, fall back to original
            return original_loss_labels(self, outputs, targets, indices, num_boxes, log)

    import types
    criterion.loss_labels = types.MethodType(weighted_loss_labels, criterion)

_orig_interpolate_pos_encoding = (
    _dinov2_mod.WindowedDinov2WithRegistersEmbeddings.interpolate_pos_encoding
)


def _interpolate_pos_encoding_compile_safe(self, embeddings, height, width):
    import torch.nn as nn

    num_patches = embeddings.shape[1] - 1
    num_positions = self.position_embeddings.shape[1] - 1

    if num_patches == num_positions and height == width:
        return self.position_embeddings

    class_pos_embed = self.position_embeddings[:, 0]
    patch_pos_embed = self.position_embeddings[:, 1:]
    dim = embeddings.shape[-1]

    height = height // self.config.patch_size
    width = width // self.config.patch_size

    sqrt_num_positions = int(num_positions**0.5)
    patch_pos_embed = patch_pos_embed.reshape(
        1, sqrt_num_positions, sqrt_num_positions, dim
    )
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

    target_dtype = patch_pos_embed.dtype
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.to(dtype=torch.float32),
        size=(int(height), int(width)),
        mode="bicubic",
        align_corners=False,
        antialias=False,  # changed: antialias=True crashes inductor on recompilation
    ).to(dtype=target_dtype)

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


_dinov2_mod.WindowedDinov2WithRegistersEmbeddings.interpolate_pos_encoding = (
    _interpolate_pos_encoding_compile_safe
)

# ---------------------------------------------------------------------------
# Monkey-patch: eliminate graph breaks in deformable attention (single-level)
# ---------------------------------------------------------------------------
# Both gen_encoder_output_proposals and ms_deform_attn_core_pytorch iterate
# over a spatial_shapes tensor, calling .item() implicitly.  With
# projector_scale = ["P4"], L=1.  The spatial_shapes tensor is built from
# src.shape (Python ints) then round-tripped through a tensor, losing
# compile-time info.  We fix this by baking the spatial shapes as Python
# constants into closures, installed before torch.compile runs.

import rfdetr.models.ops.functions.ms_deform_attn_func as _deform_func_mod
import rfdetr.models.ops.modules.ms_deform_attn as _deform_mod
import rfdetr.models.transformer as _trans_mod


def install_compile_safe_deform_patches(spatial_h: int, spatial_w: int):
    """Install torch.compile-safe patches that bake spatial shapes as Python ints.

    Must be called after the model is built but before torch.compile.
    """

    def _gen_encoder_output_proposals_fixed(
        memory, memory_padding_mask, spatial_shapes, unsigmoid=True
    ):
        N_ = memory.shape[0]
        H_, W_ = spatial_h, spatial_w  # Python ints from closure

        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask.view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_H = torch.full((N_,), H_, device=memory.device, dtype=torch.float32)
            valid_W = torch.full((N_,), W_, device=memory.device, dtype=torch.float32)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05  # lvl=0 → 2.0**0 = 1.0

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        output_proposals = proposal
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)

        if unsigmoid:
            output_proposals = torch.log(output_proposals / (1 - output_proposals))
            if memory_padding_mask is not None:
                output_proposals = output_proposals.masked_fill(
                    memory_padding_mask.unsqueeze(-1), float("inf")
                )
            output_proposals = output_proposals.masked_fill(
                ~output_proposals_valid, float("inf")
            )
        else:
            if memory_padding_mask is not None:
                output_proposals = output_proposals.masked_fill(
                    memory_padding_mask.unsqueeze(-1), float(0)
                )
            output_proposals = output_proposals.masked_fill(
                ~output_proposals_valid, float(0)
            )

        output_memory = memory
        if memory_padding_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_padding_mask.unsqueeze(-1), float(0)
            )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

        return output_memory, output_proposals

    def _ms_deform_attn_core_fixed(
        value, value_spatial_shapes, sampling_locations, attention_weights
    ):
        B, n_heads, head_dim, _ = value.shape
        _, Len_q, _, L, P, _ = sampling_locations.shape
        H, W = spatial_h, spatial_w  # Python ints from closure

        value_l_ = value.reshape(B * n_heads, head_dim, H, W)

        sampling_grids = 2 * sampling_locations - 1
        sampling_grid_l_ = sampling_grids[:, :, :, 0].transpose(1, 2).flatten(0, 1)

        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        attention_weights = attention_weights.transpose(1, 2).reshape(
            B * n_heads, 1, Len_q, L * P
        )
        output = (
            (sampling_value_l_ * attention_weights)
            .sum(-1)
            .view(B, n_heads * head_dim, Len_q)
        )
        return output.transpose(1, 2).contiguous()

    def _ms_deform_attn_forward_fixed(
        self, query, reference_points, input_flatten, input_spatial_shapes,
        input_level_start_index, input_padding_mask=None,
    ):
        """MSDeformAttn.forward with baked spatial shapes (no asserts or tensor indexing)."""
        N, Len_q, _ = query.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )

        # Single level: offset_normalizer is just [W, H]
        offset_normalizer = torch.tensor(
            [[spatial_w, spatial_h]], dtype=query.dtype, device=query.device
        )
        if reference_points.shape[-1] == 2:
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        attention_weights = F.softmax(attention_weights, -1)

        Len_in = input_flatten.shape[1]
        value = value.transpose(1, 2).contiguous().view(
            N, self.n_heads, self.d_model // self.n_heads, Len_in
        )
        output = _ms_deform_attn_core_fixed(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

    _trans_mod.gen_encoder_output_proposals = _gen_encoder_output_proposals_fixed
    _deform_func_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_fixed
    _deform_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_fixed
    _deform_mod.MSDeformAttn.forward = _ms_deform_attn_forward_fixed


# Suppress the inductor "Online softmax" warning — it fires when the reduction
# dim is tiny (L*P ≤ 4) and falls back to a 2-pass max+exp+sum which is already
# optimal for such small reductions.  Purely cosmetic.
# import warnings

# warnings.filterwarnings("ignore", message=r".*Online softmax is disabled.*")

logger = get_logger()
_BYTES_TO_MB = 1024.0 * 1024.0


def _plot_training_curves(log_path: Path) -> None:
    """Read log.txt and save loss/mAP plots to the same directory."""
    lines = log_path.read_text().strip().splitlines()
    if not lines:
        return
    records = [json.loads(l) for l in lines]
    epochs = [r["epoch"] for r in records]

    out_dir = log_path.parent

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_loss"] for r in records], label="train")
    if "test_loss" in records[0]:
        ax.plot(epochs, [r["test_loss"] for r in records], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)

    # --- mAP plot ---
    if "test_coco_eval_bbox" in records[0]:
        fig, ax = plt.subplots(figsize=(8, 5))
        map_50_95 = [r["test_coco_eval_bbox"][0] for r in records]
        map_50 = [r["test_coco_eval_bbox"][1] for r in records]
        ax.plot(epochs, map_50_95, label="mAP@50:95")
        ax.plot(epochs, map_50, label="mAP@50")
        if "ema_test_coco_eval_bbox" in records[0]:
            ax.plot(epochs, [r["ema_test_coco_eval_bbox"][0] for r in records], label="EMA mAP@50:95", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_title("Validation mAP")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "map.png", dpi=150)
        plt.close(fig)

    # --- LR plot ---
    if "train_lr" in records[0]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [r["train_lr"] for r in records])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        fig.tight_layout()
        fig.savefig(out_dir / "lr.png", dpi=150)
        plt.close(fig)

# Default pretrained weights filename for RFDETRLarge (2026 version).
PRETRAIN_WEIGHTS = "rf-detr-xxlarge.pth"

# Per-model-size architecture configs (matches rfdetr's ModelConfig for each size)
_MODEL_ARCH = {
    "large": {
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "num_windows": 2,
        "patch_size": 16,
        "resolution": 704,
        "positional_encoding_size": 704 // 16,
        "dim_feedforward": 2048,
        "pretrain_weights": "rf-detr-large-2026.pth",
    },
    "xxlarge": {
        "encoder": "dinov2_windowed_base",
        "hidden_dim": 512,
        "dec_layers": 5,
        "sa_nheads": 16,
        "ca_nheads": 32,
        "dec_n_points": 4,
        "num_windows": 2,
        "patch_size": 20,
        "resolution": 880,
        "positional_encoding_size": 880 // 20,
        "dim_feedforward": 2048,
        "pretrain_weights": "rf-detr-xxlarge.pth",
    },
}


# ---------------------------------------------------------------------------
# Helpers ported from rfdetr.engine
# ---------------------------------------------------------------------------


def _is_cuda(device: torch.device) -> bool:
    return (
        isinstance(device, torch.device)
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_initialized()
    )


def _get_autocast_args(args):
    if not torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported):
            dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
        else:
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
    if _DEPRECATED_AMP:
        return {"enabled": args.amp, "dtype": dtype}
    return {"device_type": "cuda", "enabled": args.amp, "dtype": dtype}


# ---------------------------------------------------------------------------
# train_one_epoch  –  extracted from rfdetr.engine so we can modify it
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: LWDETR,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    num_training_steps_per_epoch=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    print_freq = args.print_freq if args is not None else 10
    start_steps = epoch * num_training_steps_per_epoch

    if _DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler("cuda", enabled=args.amp)

    optimizer.zero_grad(set_to_none=True)

    header = f"Epoch: [{epoch + 1}/{args.epochs}]"
    use_progress_bar = bool(getattr(args, "progress_bar", False))
    if use_progress_bar:
        progress_iter = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=header,
            colour="green",
            disable=not utils.is_main_process(),
        )
    else:
        progress_iter = enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        )

    for data_iter_step, (samples, targets) in progress_iter:
        it = start_steps + data_iter_step

        for callback in callbacks["on_train_batch_start"]:
            callback({"step": it, "model": model, "epoch": epoch})

        samples_tensors = samples.tensors.to(device, non_blocking=True)
        samples_mask = samples.mask.to(device, non_blocking=True)
        all_targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
        ]

        new_samples = NestedTensor(samples_tensors, samples_mask)

        with autocast(**_get_autocast_args(args)):
            outputs = model(new_samples, all_targets)
            loss_dict = criterion(outputs, all_targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
            del outputs

        scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logger.error(
                "Loss is %s, stopping training. Loss dict: %s",
                loss_value,
                loss_dict_reduced,
            )
            raise ValueError(f"Loss is {loss_value}, stopping training")

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if ema_m is not None and epoch >= 0:
            ema_m.update(model)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if use_progress_bar:
            log_dict = {
                k: meter.global_avg for k, meter in metric_logger.meters.items()
            }
            postfix = {
                "lr": f"{log_dict['lr']:.6f}",
                "class_loss": f"{log_dict['class_error']:.2f}",
                "box_loss": f"{log_dict['loss_bbox']:.2f}",
                "loss": f"{log_dict['loss']:.2f}",
            }
            if _is_cuda(device):
                postfix["max_mem"] = (
                    f"{torch.cuda.max_memory_allocated(device=device) / _BYTES_TO_MB:.0f} MB"
                )
            progress_iter.set_postfix(postfix)

    metric_logger.synchronize_between_processes()
    logger.info(f"Epoch {epoch + 1} stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Args / model building
# ---------------------------------------------------------------------------


def _make_args(
    dataset_dir: str,
    num_classes: int,
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    num_workers: int,
    model_size: str = "xxlarge",
) -> argparse.Namespace:
    """Build the args namespace that rfdetr internals expect.

    Supports "large" and "xxlarge" model sizes. Only the values that
    actually vary are parameters; everything else is pinned.
    """
    arch = _MODEL_ARCH[model_size]
    return argparse.Namespace(
        # --- Model architecture ---
        encoder=arch["encoder"],
        hidden_dim=arch["hidden_dim"],
        dec_layers=arch["dec_layers"],
        sa_nheads=arch["sa_nheads"],
        ca_nheads=arch["ca_nheads"],
        dec_n_points=arch["dec_n_points"],
        num_windows=arch["num_windows"],
        patch_size=arch["patch_size"],
        projector_scale=["P4"],
        out_feature_indexes=[3, 6, 9, 12],
        positional_encoding_size=arch["positional_encoding_size"],
        resolution=arch["resolution"],
        num_classes=num_classes,
        num_queries=300,
        group_detr=13,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        layer_norm=True,
        rms_norm=False,
        amp=True,
        aux_loss=True,
        position_embedding="sine",
        use_cls_token=False,
        vit_encoder_num_layers=12,
        pretrained_encoder=None,
        window_block_indexes=None,
        freeze_encoder=False,
        backbone_lora=False,
        force_no_pretrain=False,
        gradient_checkpointing=False,
        dim_feedforward=arch["dim_feedforward"],
        decoder_norm="LN",
        freeze_batch_norm=False,
        segmentation_head=False,
        mask_downsample_ratio=4,
        pretrain_weights=arch["pretrain_weights"],
        pretrain_exclude_keys=None,
        pretrain_keys_modify_to_load=None,
        encoder_only=False,
        backbone_only=False,
        # --- Training hyper-parameters ---
        lr=lr,
        lr_encoder=1e-5,
        weight_decay=1e-4,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        epochs=epochs,
        clip_max_norm=0.1,
        lr_vit_layer_decay=0.8,
        lr_component_decay=0.7,
        # --- LR scheduler ---
        lr_scheduler="cosine",
        lr_drop=100,
        lr_min_factor=0.01,
        warmup_epochs=3.0,
        # --- Drop ---
        dropout=0,
        drop_path=0.0,
        drop_mode="standard",
        drop_schedule="constant",
        cutoff_epoch=0,
        # --- Loss ---
        set_cost_class=2,
        set_cost_bbox=5,
        set_cost_giou=2,
        cls_loss_coef=1.0,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        focal_alpha=0.25,
        ia_bce_loss=True,
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        # --- EMA ---
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        # --- Dataset ---
        dataset_file="roboflow",
        dataset_dir=dataset_dir,
        coco_path=None,
        square_resize_div_64=True,
        multi_scale=False,
        expanded_scales=True,
        do_random_resize_via_padding=False,
        aug_config=AUG_CONFIG,
        # --- Output / misc ---
        output_dir=output_dir,
        dont_save_weights=False,
        checkpoint_interval=10,
        num_select=300,
        eval_max_dets=500,
        device="cuda",
        seed=42,
        resume="",
        start_epoch=0,
        eval=False,
        num_workers=num_workers,
        distributed=False,
        run_test=False,
        print_freq=10,
        progress_bar=True,
        fp16_eval=False,
        do_benchmark=False,
        world_size=1,
    )


def _build_and_load_model(
    args: argparse.Namespace,
    pretrain_weights: str,
) -> torch.nn.Module:
    """Build an LWDETR model, download pretrained weights, and load them.

    The model is first built with ``num_classes=90`` (matching the pretrained
    checkpoint), the checkpoint is loaded, and then the detection head is
    re-initialised to ``args.num_classes`` by tiling the pretrained weights.
    """
    actual_num_classes = args.num_classes

    # Build with the pretrained checkpoint's class count so the state-dict
    # keys match perfectly.
    args.num_classes = 90
    model = build_model(args)
    args.num_classes = actual_num_classes

    # Download & validate pretrained weights.
    download_pretrain_weights(pretrain_weights)
    validate_pretrain_weights(pretrain_weights, strict=False)
    checkpoint = torch.load(pretrain_weights, map_location="cpu", weights_only=False)

    # Trim query embeddings to match group_detr config.
    num_desired_queries = args.num_queries * args.group_detr
    for name in ("refpoint_embed.weight", "query_feat.weight"):
        if name in checkpoint["model"]:
            checkpoint["model"][name] = checkpoint["model"][name][:num_desired_queries]

    model.load_state_dict(checkpoint["model"], strict=False)

    # Reinitialise the detection head from 91 → actual_num_classes+1 by
    # tiling the pretrained class-embed weights.
    model.reinitialize_detection_head(actual_num_classes + 1)

    return model



def train(
    dataset_dir: Path,
    output_dir: str = "output",
    epochs: int = 20,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lr: float = 1e-4,
    num_workers: int = 4,
    # --- Full-image competition eval ---
    val_coco: dict | None = None,
    val_images_dir: Path | None = None,
    competition_eval_interval: int = 5,
    # --- Wandb viz ---
    wandb_viz_interval: int = 5,
    wandb_viz_n_images: int = 8,
    # --- Model ---
    model_size: str = "xxlarge",
    # --- Run name ---
    run_name: str = "",
    # --- Crop mode ---
    crop_mode: str = "fixed",
    # --- Per-class loss reweighting ---
    class_weight_loss: bool = False,
    # --- Copy-paste augmentation ---
    copy_paste: bool = False,
    # --- Label merging ---
    merge_labels: bool = False,
    # --- Max image dimension ---
    max_image_dim: int = 0,
    # --- Crop scale range ---
    crop_scale_min: float = 0.5,
    crop_scale_max: float = 1.5,
    # --- Synthetic data ---
    synthetic_data: str = "",
    synthetic_weight: float = 0.0,
) -> torch.nn.Module:
    """Train an LWDETR model directly, bypassing the RFDETR wrapper classes.

    This is functionally equivalent to ``pipeline.train`` but operates on the
    raw ``nn.Module`` so the training loop is fully visible and extensible.
    Returns the best model (EMA or regular).
    """
    device = torch.device("cuda")

    # --- Resolve num_classes from dataset ----------------------------------
    class_names = RFDETR._load_classes(str(dataset_dir))
    num_classes = len(class_names) + 1  # roboflow convention

    # --- Args namespace used by rfdetr helpers -----------------------------
    args = _make_args(
        dataset_dir=str(dataset_dir),
        num_classes=num_classes,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        num_workers=num_workers,
        model_size=model_size,
    )

    # --- W&B ---------------------------------------------------------------
    wandb.init(
        entity="ankile",
        project="norgesgruppen-rfdetr",
        name=run_name or f"{model_size}_lr={lr}_bs={batch_size*grad_accum_steps}_ep={epochs}",
        config={
            # Optimizer
            "lr": lr,
            "lr_encoder": args.lr_encoder,
            "lr_scheduler": args.lr_scheduler,
            "lr_vit_layer_decay": args.lr_vit_layer_decay,
            "lr_component_decay": args.lr_component_decay,
            "lr_drop": args.lr_drop,
            "lr_min_factor": args.lr_min_factor,
            "weight_decay": args.weight_decay,
            "clip_max_norm": args.clip_max_norm,
            "warmup_epochs": args.warmup_epochs,
            # Training
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "seed": args.seed,
            # Model
            "model_size": model_size,
            "resolution": args.resolution,
            "num_classes": num_classes,
            "num_queries": args.num_queries,
            "hidden_dim": args.hidden_dim,
            "dec_layers": args.dec_layers,
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay,
            # Data
            "dataset_dir": str(dataset_dir),
            "output_dir": output_dir,
            "num_workers": num_workers,
            "crop_mode": crop_mode,
            "max_image_dim": max_image_dim,
            "copy_paste": copy_paste,
            "synthetic_data": synthetic_data,
            "synthetic_weight": synthetic_weight,
            # Competition eval
            "full_image_eval": val_coco is not None,
            "n_val_images": len(val_coco["images"]) if val_coco else 0,
            "competition_eval_interval": competition_eval_interval,
            "wandb_viz_interval": wandb_viz_interval,
            # Loss weights
            "cls_loss_coef": args.cls_loss_coef,
            "bbox_loss_coef": args.bbox_loss_coef,
            "giou_loss_coef": args.giou_loss_coef,
            "set_cost_class": args.set_cost_class,
            "set_cost_bbox": args.set_cost_bbox,
            "set_cost_giou": args.set_cost_giou,
            # Exclusions & merges
            "excluded_image_ids": sorted(EXCLUDED_IMAGE_IDS),
            "merge_labels": merge_labels,
            "label_merge_map": {str(k): v for k, v in LABEL_MERGE_MAP.items()} if merge_labels else {},
            # Host
            "hostname": socket.gethostname(),
        },
    )

    # --- Seed --------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- Model -------------------------------------------------------------
    model = _build_and_load_model(args, args.pretrain_weights)
    model.to(device)

    # --- Criterion & post-processor ----------------------------------------
    criterion, postprocess = build_criterion_and_postprocessors(args)

    # --- Per-class loss reweighting ----------------------------------------
    if class_weight_loss:
        # pred_logits shape is [B, Q, num_classes+1] (criterion.num_classes)
        logit_dim = num_classes + 1  # 358 for our 356-category dataset
        class_weights = compute_class_weights(args.dataset_dir, logit_dim)
        _install_class_weighted_loss(criterion, class_weights)
        logger.info(
            "Per-class loss weights: min=%.2f, max=%.2f, median=%.2f (num_classes=%d)",
            class_weights.min().item(), class_weights.max().item(),
            class_weights.median().item(), num_classes,
        )
    else:
        logger.info("Per-class loss reweighting: OFF")

    # --- Optimizer ---------------------------------------------------------
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Number of trainable parameters: %d (%.2f M)", n_parameters, n_parameters / 1e6
    )

    param_dicts = get_param_dict(args, model)
    param_dicts = [p for p in param_dicts if p["params"].requires_grad]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Datasets & dataloaders --------------------------------------------
    if crop_mode == "random":
        from norgesgruppen.random_crop_dataset import RandomCropDataset
        # Load the train COCO JSON (written by prepare_split_datasets)
        import json as _json
        with open(Path(args.dataset_dir) / "train" / "_annotations.coco.json") as f:
            train_coco_dict = _json.load(f)
        dataset_train = RandomCropDataset(
            train_coco_dict,
            resolution=args.resolution,
            scale_range=(crop_scale_min, crop_scale_max),
            samples_per_epoch=5000,
            copy_paste=copy_paste,
            max_image_dim=max_image_dim,
        )
        logger.info("Using RandomCropDataset: %d samples/epoch, scale_range=(%.1f, %.1f), copy_paste=%s",
                     len(dataset_train), crop_scale_min, crop_scale_max, copy_paste)

        # Optionally interleave with synthetic data
        if synthetic_data:
            import json as _json_syn
            with open(Path(synthetic_data) / "annotations.json") as f:
                syn_coco = _json_syn.load(f)
            dataset_synthetic = RandomCropDataset(
                syn_coco,
                resolution=args.resolution,
                scale_range=(crop_scale_min, crop_scale_max),
                samples_per_epoch=5000,
                copy_paste=False,
                max_image_dim=max_image_dim,
            )
            dataset_train = InterleavedDataset(
                dataset_real=dataset_train,
                dataset_synthetic=dataset_synthetic,
                synthetic_weight=synthetic_weight,
                samples_per_epoch=5000,
            )
            logger.info(
                "Synthetic data: %d images from %s, weight=%.2f",
                len(syn_coco["images"]), synthetic_data, synthetic_weight,
            )
    else:
        dataset_train = build_dataset(
            image_set="train", args=args, resolution=args.resolution
        )
    # has_val: whether rfdetr's built-in COCO eval runs (on the dummy 1-image valid set).
    # The real evaluation uses full-image competition eval via val_coco.
    has_val = True  # dummy valid always exists from prepare_split_datasets
    if has_val:
        dataset_val = build_dataset(image_set="val", args=args, resolution=args.resolution)
        logger.info(
            "Dataset loaded: %d training, %d validation",
            len(dataset_train),
            len(dataset_val),
        )
    else:
        logger.info("Dataset loaded: %d training, no validation", len(dataset_train))

    effective_batch_size = args.batch_size * args.grad_accum_steps
    min_batches = 5

    if len(dataset_train) < effective_batch_size * min_batches:
        sampler_train = torch.utils.data.RandomSampler(
            dataset_train,
            replacement=True,
            num_samples=effective_batch_size * min_batches,
        )
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=effective_batch_size,
            collate_fn=utils.collate_fn,
            num_workers=num_workers,
            sampler=sampler_train,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, effective_batch_size, drop_last=True
        )
        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    if has_val:
        data_loader_val = DataLoader(
            dataset_val,
            args.batch_size,
            sampler=torch.utils.data.SequentialSampler(dataset_val),
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        data_loader_val = None
        base_ds = None

    # --- LR scheduler ------------------------------------------------------
    total_batch_size_for_lr = args.batch_size * args.grad_accum_steps
    num_steps_per_epoch_lr = (
        len(dataset_train) + total_batch_size_for_lr - 1
    ) // total_batch_size_for_lr
    total_steps_lr = num_steps_per_epoch_lr * args.epochs
    warmup_steps = num_steps_per_epoch_lr * args.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if args.lr_scheduler == "cosine":
            progress = float(step - warmup_steps) / float(
                max(1, total_steps_lr - warmup_steps)
            )
            return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        # step scheduler
        if step < args.lr_drop * num_steps_per_epoch_lr:
            return 1.0
        return 0.1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- EMA ---------------------------------------------------------------
    ema_m = (
        ModelEma(model, decay=args.ema_decay, tau=args.ema_tau)
        if args.use_ema
        else None
    )

    # --- torch.compile (after optimizer + EMA setup) -----------------------
    model_without_compile = model
    if not args.gradient_checkpointing:
        spatial_side = args.resolution // args.patch_size
        install_compile_safe_deform_patches(spatial_side, spatial_side)
        model = torch.compile(model)
    else:
        logger.info("Skipping torch.compile (incompatible with gradient checkpointing)")

    # --- Training bookkeeping ----------------------------------------------
    output_path = Path(args.output_dir)
    num_training_steps_per_epoch = (
        len(dataset_train) + effective_batch_size - 1
    ) // effective_batch_size
    callbacks: defaultdict[str, list] = defaultdict(list)

    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    best_map_5095 = 0.0
    best_map_ema_5095 = 0.0
    best_competition_score = 0.0

    def _should_run_full_image_eval(epoch: int) -> bool:
        """Decide if full-image competition eval should run this epoch."""
        if val_coco is None or val_images_dir is None:
            return False
        if epoch == args.epochs - 1:  # always on last epoch
            return True
        return (epoch + 1) % competition_eval_interval == 0

    # --- Encoder freeze for initial warmup epochs ---------------------------
    unfreeze_epoch = int(math.ceil(args.warmup_epochs))
    encoder = model_without_compile.backbone[0].encoder
    for p in encoder.parameters():
        p.requires_grad = False
    logger.info("Encoder frozen for first %d epoch(s)", unfreeze_epoch)

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # Unfreeze encoder after warmup
        if epoch == unfreeze_epoch:
            for p in encoder.parameters():
                p.requires_grad = True
            logger.info("Encoder unfrozen at epoch %d", epoch)

        epoch_start_time = time.time()

        model.train()
        criterion.train()

        train_stats = train_one_epoch(
            model,
            criterion,
            lr_scheduler,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            ema_m=ema_m,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            args=args,
            callbacks=callbacks,
        )

        train_epoch_time_str = str(
            datetime.timedelta(seconds=int(time.time() - epoch_start_time))
        )

        # --- Save checkpoint -----------------------------------------------
        if args.output_dir:
            checkpoint_paths = [output_path / "checkpoint.pth"]
            if (epoch + 1) % args.lr_drop == 0 or (
                epoch + 1
            ) % args.checkpoint_interval == 0:
                checkpoint_paths.append(output_path / f"checkpoint{epoch:04}.pth")
            for cp in checkpoint_paths:
                weights = {
                    "model": model_without_compile.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if ema_m is not None:
                    weights["ema_model"] = ema_m.module.state_dict()
                cp.parent.mkdir(parents=True, exist_ok=True)
                save_on_master(weights, cp)

        # --- Evaluate (regular model) --------------------------------------
        test_stats = {}
        coco_evaluator = None
        if has_val:
            with torch.inference_mode():
                test_stats, coco_evaluator = evaluate(
                    model,
                    criterion,
                    postprocess,
                    data_loader_val,
                    base_ds,
                    device,
                    args=args,
                    header="Test",
                )

            map_regular = test_stats["coco_eval_bbox"][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                best_map_5095 = max(best_map_5095, map_regular)
                save_on_master(
                    {
                        "model": model_without_compile.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    output_path / "checkpoint_best_regular.pth",
                )

        # --- Full-image competition eval (matches submission behavior) -----
        full_comp_score = None
        if _should_run_full_image_eval(epoch):
            from norgesgruppen.competition_eval import (
                evaluate_competition_metric,
                generate_analysis_plots,
                render_val_predictions,
            )
            eval_model = ema_m.module if ema_m is not None else model_without_compile
            full_comp_score, eval_gts, eval_preds = evaluate_competition_metric(
                eval_model, postprocess, val_coco, val_images_dir,
                resolution=args.resolution, threshold=0.2,
                max_image_dim=max_image_dim,
            )
            logger.info("Competition score (full-image, %s): %s",
                        "EMA" if ema_m is not None else "regular", full_comp_score)

            if full_comp_score.combined > best_competition_score:
                best_competition_score = full_comp_score.combined
                save_on_master(
                    {
                        "model": eval_model.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "competition_score": full_comp_score.combined,
                    },
                    output_path / "checkpoint_best_competition.pth",
                )
                logger.info("New best competition score: %.4f", best_competition_score)

            # Wandb visualizations + analysis plots
            if (epoch + 1) % wandb_viz_interval == 0 or epoch == args.epochs - 1:
                viz_images = render_val_predictions(
                    eval_model, postprocess, val_coco, val_images_dir,
                    n_images=wandb_viz_n_images, resolution=args.resolution,
                    max_image_dim=max_image_dim,
                )
                wandb.log({"val_predictions": viz_images}, step=epoch)

                analysis_plots = generate_analysis_plots(eval_gts, eval_preds, val_coco)
                wandb.log(analysis_plots, step=epoch)

        log_stats: dict = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        # --- Evaluate (EMA model) ------------------------------------------
        if has_val and ema_m is not None:
            ema_test_stats, _ = evaluate(
                ema_m.module,
                criterion,
                postprocess,
                data_loader_val,
                base_ds,
                device,
                args=args,
                header="Test-ema",
            )
            log_stats.update({f"ema_test_{k}": v for k, v in ema_test_stats.items()})
            map_ema = ema_test_stats["coco_eval_bbox"][0]
            best_map_ema_5095 = max(best_map_ema_5095, map_ema)
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                save_on_master(
                    {
                        "model": ema_m.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    output_path / "checkpoint_best_ema.pth",
                )

        log_stats.update(best_map_holder.summary())
        log_stats["train_epoch_time"] = train_epoch_time_str
        log_stats["epoch_time"] = str(
            datetime.timedelta(seconds=int(time.time() - epoch_start_time))
        )

        # --- W&B logging (namespaced for panel grouping) ----------------------
        wandb_stats = {}

        # eval/ — Full-image competition scores (periodic, matches submission)
        if full_comp_score is not None:
            wandb_stats["eval/hybrid_score"] = full_comp_score.combined
            wandb_stats["eval/detection_map"] = full_comp_score.detection_map
            wandb_stats["eval/classification_map"] = full_comp_score.classification_map
            wandb_stats["eval/best_competition_score"] = best_competition_score

        # train/ — Training health
        for k in ("loss", "lr", "class_error"):
            if k in train_stats:
                wandb_stats[f"train/{k}"] = train_stats[k]

        # coco/ — COCO mAP
        for log_prefix, ns in [("test_", ""), ("ema_test_", "ema_")]:
            bbox_key = f"{log_prefix}coco_eval_bbox"
            if bbox_key in log_stats:
                vals = log_stats[bbox_key]
                wandb_stats[f"coco/{ns}mAP_50_95"] = vals[0]
                wandb_stats[f"coco/{ns}mAP_50"] = vals[1]
                wandb_stats[f"coco/{ns}mAP_75"] = vals[2]

        # loss/ — Per-component loss breakdown (final decoder layer)
        _loss_sources = [("train", train_stats), ("val", test_stats)]
        if has_val and ema_m is not None:
            _loss_sources.append(("ema_val", ema_test_stats))
        for prefix, stats in _loss_sources:
            for lk in ("loss_ce", "loss_bbox", "loss_giou"):
                if lk in stats and isinstance(stats[lk], (int, float)):
                    wandb_stats[f"loss/{prefix}_{lk}"] = stats[lk]

        # aux/ — Auxiliary decoder layer & encoder losses (skip unscaled)
        _aux_suffixes = tuple(f"_{i}" for i in range(10)) + ("_enc",)
        for prefix, stats in _loss_sources:
            for k, v in stats.items():
                if not isinstance(v, (int, float)):
                    continue
                if "unscaled" in k:
                    continue
                if any(k.endswith(s) for s in _aux_suffixes):
                    wandb_stats[f"aux/{prefix}_{k}"] = v

        wandb.log(wandb_stats, step=epoch)

        # --- Write logs & eval artifacts -----------------------------------
        if args.output_dir and is_main_process():
            with (output_path / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            _plot_training_curves(output_path / "log.txt")
            if coco_evaluator is not None:
                (output_path / "eval").mkdir(exist_ok=True)
                filenames = ["latest.pth"]
                if epoch % 50 == 0:
                    filenames.append(f"{epoch:03}.pth")
                for name in filenames:
                    torch.save(
                        coco_evaluator.coco_eval["bbox"].eval,
                        output_path / "eval" / name,
                    )

    # --- Finalise: pick best model and write results -----------------------
    best_is_ema = best_map_ema_5095 > best_map_5095

    if is_main_process():
        if has_val:
            # Prefer competition-metric best if available
            comp_ckpt = output_path / "checkpoint_best_competition.pth"
            if comp_ckpt.exists():
                shutil.copy2(comp_ckpt, output_path / "checkpoint_best_total.pth")
                utils.strip_checkpoint(output_path / "checkpoint_best_total.pth")
                logger.info("Best total model: competition metric (score %.4f)", best_competition_score)
            else:
                src = output_path / (
                    "checkpoint_best_ema.pth" if best_is_ema else "checkpoint_best_regular.pth"
                )
                if src.exists():
                    shutil.copy2(src, output_path / "checkpoint_best_total.pth")
                    utils.strip_checkpoint(output_path / "checkpoint_best_total.pth")

            results = (ema_test_stats if best_is_ema else test_stats)["results_json"]
            results["class_map"] = {"valid": results["class_map"]}
            with open(output_path / "results.json", "w") as f:
                json.dump(results, f)
            logger.info("Results saved to %s", output_path / "results.json")

        total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info("Training time %s", total_time_str)

    wandb.finish()

    if best_is_ema and ema_m is not None:
        return ema_m.module
    return model_without_compile

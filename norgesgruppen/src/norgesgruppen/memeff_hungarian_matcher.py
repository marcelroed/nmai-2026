"""
Modules to compute the matching cost and solve the corresponding LSAP.

Monkey patches these with my own implementation below.
"""

import numpy as np
import rfdetr.models.lwdetr as lwdetr
import rfdetr.models.matcher as matcher
import torch
import torch.nn.functional as F
from rfdetr.models.segmentation_head import point_sample
from rfdetr.util.box_ops import (
    batch_dice_loss,
    batch_sigmoid_ce_loss,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from scipy.optimize import linear_sum_assignment
from torch import nn


def _aligned_iou_from_cxcywh(
    src_boxes: torch.Tensor, tgt_boxes: torch.Tensor
) -> torch.Tensor:
    """IoU between aligned pairs of boxes, shape [N]."""
    if src_boxes.numel() == 0:
        return src_boxes.new_zeros((0,))

    src_xyxy = box_cxcywh_to_xyxy(src_boxes.detach())
    tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)

    lt = torch.max(src_xyxy[:, :2], tgt_xyxy[:, :2])
    rb = torch.min(src_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    src_area = (src_xyxy[:, 2] - src_xyxy[:, 0]).clamp(min=0) * (
        src_xyxy[:, 3] - src_xyxy[:, 1]
    ).clamp(min=0)
    tgt_area = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0) * (
        tgt_xyxy[:, 3] - tgt_xyxy[:, 1]
    ).clamp(min=0)
    union = src_area + tgt_area - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _aligned_giou_from_cxcywh(
    src_boxes: torch.Tensor, tgt_boxes: torch.Tensor
) -> torch.Tensor:
    """Generalized IoU between aligned pairs of boxes, shape [N]."""
    if src_boxes.numel() == 0:
        return src_boxes.new_zeros((0,))

    src_xyxy = box_cxcywh_to_xyxy(src_boxes)
    tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)

    lt_inter = torch.max(src_xyxy[:, :2], tgt_xyxy[:, :2])
    rb_inter = torch.min(src_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0)
    inter = wh_inter[:, 0] * wh_inter[:, 1]

    src_area = (src_xyxy[:, 2] - src_xyxy[:, 0]).clamp(min=0) * (
        src_xyxy[:, 3] - src_xyxy[:, 1]
    ).clamp(min=0)
    tgt_area = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0) * (
        tgt_xyxy[:, 3] - tgt_xyxy[:, 1]
    ).clamp(min=0)
    union = src_area + tgt_area - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

    lt_enclose = torch.min(src_xyxy[:, :2], tgt_xyxy[:, :2])
    rb_enclose = torch.max(src_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh_enclose = (rb_enclose - lt_enclose).clamp(min=0)
    area_enclose = wh_enclose[:, 0] * wh_enclose[:, 1]
    return iou - torch.where(
        area_enclose > 0,
        (area_enclose - union) / area_enclose,
        torch.zeros_like(area_enclose),
    )


def _loss_labels_memeff(self, outputs, targets, indices, num_boxes, log=True):
    """Memory-efficient replacement for SetCriterion.loss_labels."""
    assert "pred_logits" in outputs
    src_logits = outputs["pred_logits"]

    idx = self._get_src_permutation_idx(indices)
    target_classes_o = torch.cat(
        [t["labels"][j] for t, (_, j) in zip(targets, indices)]
    )

    if self.ia_bce_loss:
        alpha = self.focal_alpha
        gamma = 2
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        pos_ious = _aligned_iou_from_cxcywh(src_boxes, target_boxes).detach()

        prob = src_logits.sigmoid()
        pos_weights = torch.zeros_like(src_logits)
        neg_weights = prob**gamma

        pos_ind = [x for x in idx]
        pos_ind.append(target_classes_o)
        t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
        t = torch.clamp(t, 0.01).detach()

        pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
        neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)
        loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (
            pos_weights + neg_weights
        )
        loss_ce = loss_ce.sum() / num_boxes

    elif self.use_position_supervised_loss:
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        pos_ious = _aligned_iou_from_cxcywh(src_boxes, target_boxes).detach()
        pos_ious_func = pos_ious

        cls_iou_func_targets = torch.zeros(
            (src_logits.shape[0], src_logits.shape[1], self.num_classes),
            dtype=src_logits.dtype,
            device=src_logits.device,
        )

        pos_ind = [x for x in idx]
        pos_ind.append(target_classes_o)
        cls_iou_func_targets[tuple(pos_ind)] = pos_ious_func.to(
            cls_iou_func_targets.dtype
        )
        norm_cls_iou_func_targets = cls_iou_func_targets / (
            cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(
                1, True
            )
            + 1e-8
        )
        loss_ce = (
            lwdetr.position_supervised_loss(
                src_logits,
                norm_cls_iou_func_targets,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )

    elif self.use_varifocal_loss:
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        pos_ious = _aligned_iou_from_cxcywh(src_boxes, target_boxes).detach()

        cls_iou_targets = torch.zeros(
            (src_logits.shape[0], src_logits.shape[1], self.num_classes),
            dtype=src_logits.dtype,
            device=src_logits.device,
        )

        pos_ind = [x for x in idx]
        pos_ind.append(target_classes_o)
        cls_iou_targets[tuple(pos_ind)] = pos_ious
        loss_ce = (
            lwdetr.sigmoid_varifocal_loss(
                src_logits,
                cls_iou_targets,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
    else:
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            lwdetr.sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )

    losses = {"loss_ce": loss_ce}
    if log:
        losses["class_error"] = (
            100 - lwdetr.accuracy(src_logits[idx], target_classes_o)[0]
        )
    return losses


def _loss_boxes_memeff(self, outputs, targets, indices, num_boxes):
    """Memory-efficient replacement for SetCriterion.loss_boxes."""
    assert "pred_boxes" in outputs
    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = torch.cat(
        [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
    )

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
    losses = {"loss_bbox": loss_bbox.sum() / num_boxes}

    loss_giou = 1 - _aligned_giou_from_cxcywh(src_boxes, target_boxes)
    losses["loss_giou"] = loss_giou.sum() / num_boxes
    return losses


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).

    Cost matrices are computed per batch element to avoid quadratic growth in comparisons
    as batch size increases.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        use_pos_only: bool = False,
        use_position_modulated_cost: bool = False,
        mask_point_sample_ratio: int = 16,
        cost_mask_ce: float = 1,
        cost_mask_dice: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: Relative weight of the classification error in the matching cost
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: Relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs can't be 0"
        )
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice

    def _compute_mask_costs(
        self, outputs, b_idx, num_queries, tgt_masks_b, point_coords
    ):
        """Compute mask CE and Dice costs for a single batch element.

        Args:
            outputs: Full model outputs dict.
            b_idx: Batch index.
            num_queries: Number of queries per image.
            tgt_masks_b: Target masks for this batch element [num_targets, H, W].
            point_coords: Sampled point coordinates [1, num_points, 2].

        Returns:
            cost_mask_ce: [num_queries, num_targets] cross-entropy cost.
            cost_mask_dice: [num_queries, num_targets] dice cost.
        """
        if isinstance(outputs["pred_masks"], torch.Tensor):
            out_masks_b = outputs["pred_masks"][b_idx]  # [num_queries, H, W]
            pred_masks_logits = point_sample(
                out_masks_b.unsqueeze(1),
                point_coords.repeat(out_masks_b.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
        else:
            spatial_features = outputs["pred_masks"]["spatial_features"]
            query_features = outputs["pred_masks"]["query_features"]
            bias = outputs["pred_masks"]["bias"]

            spatial_b = spatial_features[b_idx].unsqueeze(0)  # [1, C, H, W]
            query_b = query_features[b_idx].unsqueeze(0)  # [1, num_queries, C]
            bias_b = bias[b_idx].unsqueeze(0)  # [1, num_queries, 1]

            sampled = point_sample(
                spatial_b,
                point_coords.repeat(spatial_b.shape[0], 1, 1),
                align_corners=False,
            )  # [1, C, num_points]
            pred_masks_logits = torch.einsum("bcp,bnc->bnp", sampled, query_b) + bias_b
            pred_masks_logits = pred_masks_logits.squeeze(
                0
            )  # [num_queries, num_points]

        tgt_masks_b = tgt_masks_b.to(pred_masks_logits.dtype)
        tgt_masks_flat = point_sample(
            tgt_masks_b.unsqueeze(1),
            point_coords.repeat(tgt_masks_b.shape[0], 1, 1),
            align_corners=False,
            mode="nearest",
        ).squeeze(1)  # [num_targets, num_points]

        cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)
        cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        return cost_mask_ce, cost_mask_dice

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """Performs the matching

        Params:
            outputs: Dict containing at least:
                 "pred_logits": Tensor [batch_size, num_queries, num_classes]
                 "pred_boxes": Tensor [batch_size, num_queries, 4]
            targets: List of dicts (len = batch_size), each containing:
                 "labels": Tensor [num_target_boxes]
                 "boxes": Tensor [num_target_boxes, 4]
                 "masks": (optional) Tensor [num_target_boxes, H, W]
            group_detr: Number of groups used for matching.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        masks_present = "masks" in targets[0]

        # Precompute point_coords once if masks are present
        point_coords = None
        if masks_present:
            if isinstance(outputs["pred_masks"], torch.Tensor):
                mask_h, mask_w = outputs["pred_masks"].shape[-2:]
            else:
                spatial_features = outputs["pred_masks"]["spatial_features"]
                mask_h, mask_w = spatial_features.shape[-2:]
            num_points = mask_h * mask_w // self.mask_point_sample_ratio
            point_coords = torch.rand(
                1,
                num_points,
                2,
                device=outputs["pred_logits"].device,
            )

        alpha = 0.25
        gamma = 2.0
        g_num_queries = num_queries // group_detr

        indices = [None] * bs

        # Phase 1: Compute all cost matrices on GPU (no sync)
        cost_matrices_gpu = [None] * bs
        num_targets_list = []

        for b_idx in range(bs):
            tgt_ids_b = targets[b_idx]["labels"]
            tgt_bbox_b = targets[b_idx]["boxes"]
            num_targets_b = tgt_ids_b.shape[0]
            num_targets_list.append(num_targets_b)

            if num_targets_b == 0:
                indices[b_idx] = (
                    torch.as_tensor([], dtype=torch.int64),
                    torch.as_tensor([], dtype=torch.int64),
                )
                continue

            pred_logits_b = outputs["pred_logits"][b_idx]
            out_prob_b = pred_logits_b.sigmoid()
            out_bbox_b = outputs["pred_boxes"][b_idx]

            neg_cost_class = (
                (1 - alpha) * (out_prob_b**gamma) * (-F.logsigmoid(-pred_logits_b))
            )
            pos_cost_class = (
                alpha * ((1 - out_prob_b) ** gamma) * (-F.logsigmoid(pred_logits_b))
            )
            cost_class = pos_cost_class[:, tgt_ids_b] - neg_cost_class[:, tgt_ids_b]

            cost_bbox = torch.cdist(out_bbox_b, tgt_bbox_b, p=1)

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox_b),
                box_cxcywh_to_xyxy(tgt_bbox_b),
            )
            cost_giou = -giou

            C_b = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )

            if masks_present:
                cost_mask_ce, cost_mask_dice = self._compute_mask_costs(
                    outputs,
                    b_idx,
                    num_queries,
                    targets[b_idx]["masks"],
                    point_coords,
                )
                C_b = (
                    C_b
                    + self.cost_mask_ce * cost_mask_ce
                    + self.cost_mask_dice * cost_mask_dice
                )

            # Clean NaN/Inf on GPU
            C_b = C_b.float()
            if C_b.numel() > 0:
                finite_mask = C_b.isfinite()
                max_cost = C_b.where(finite_mask, C_b.new_tensor(0.0)).max()
                C_b = C_b.where(finite_mask, max_cost * 2)

            cost_matrices_gpu[b_idx] = C_b

        # Phase 2: Pad cost matrices to same width and batch-transfer to CPU.
        # This avoids one GPU sync per batch element.
        active = [(b, C) for b, C in enumerate(cost_matrices_gpu) if C is not None]
        if active:
            max_targets = max(C.shape[1] for _, C in active)
            padded = torch.stack(
                [
                    F.pad(C, (0, max_targets - C.shape[1]), value=1e9)
                    for _, C in active
                ]
            )  # [num_active, num_queries, max_targets]
            padded_cpu = padded.cpu()  # single sync point

            for local_idx, (b_idx, _) in enumerate(active):
                nt = num_targets_list[b_idx]
                C_b = padded_cpu[local_idx, :, :nt]

                C_groups = C_b.split(g_num_queries, dim=0)
                matched_pred = []
                matched_tgt = []
                for g_i, C_g in enumerate(C_groups):
                    row_ind, col_ind = linear_sum_assignment(C_g.numpy())
                    matched_pred.append(row_ind + g_i * g_num_queries)
                    matched_tgt.append(col_ind)

                indices[b_idx] = (
                    torch.as_tensor(np.concatenate(matched_pred), dtype=torch.int64),
                    torch.as_tensor(np.concatenate(matched_tgt), dtype=torch.int64),
                )

        return indices


    @torch.no_grad()
    def forward_multi(self, output_dicts, targets, group_detr=1):
        """Match multiple output layers (main + aux + enc) with a single GPU→CPU sync.

        Args:
            output_dicts: List of output dicts, each with pred_logits and pred_boxes.
            targets: Same targets for all layers.
            group_detr: Number of groups.

        Returns:
            List of indices lists, one per output dict.
        """
        num_layers = len(output_dicts)
        bs, num_queries = output_dicts[0]["pred_logits"].shape[:2]
        g_num_queries = num_queries // group_detr

        alpha = 0.25
        gamma = 2.0

        # Phase 1: compute all cost matrices on GPU
        # cost_info[layer_idx][b_idx] = (C_b_gpu, num_targets_b) or None
        cost_info = [[None] * bs for _ in range(num_layers)]
        all_empty_indices = {}  # (layer, b_idx) -> empty indices

        for layer_idx, out in enumerate(output_dicts):
            for b_idx in range(bs):
                tgt_ids_b = targets[b_idx]["labels"]
                tgt_bbox_b = targets[b_idx]["boxes"]
                num_targets_b = tgt_ids_b.shape[0]

                if num_targets_b == 0:
                    all_empty_indices[(layer_idx, b_idx)] = (
                        torch.as_tensor([], dtype=torch.int64),
                        torch.as_tensor([], dtype=torch.int64),
                    )
                    continue

                pred_logits_b = out["pred_logits"][b_idx]
                out_prob_b = pred_logits_b.sigmoid()
                out_bbox_b = out["pred_boxes"][b_idx]

                neg_cost_class = (
                    (1 - alpha) * (out_prob_b**gamma) * (-F.logsigmoid(-pred_logits_b))
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob_b) ** gamma) * (-F.logsigmoid(pred_logits_b))
                )
                cost_class = pos_cost_class[:, tgt_ids_b] - neg_cost_class[:, tgt_ids_b]
                cost_bbox = torch.cdist(out_bbox_b, tgt_bbox_b, p=1)
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox_b),
                    box_cxcywh_to_xyxy(tgt_bbox_b),
                )

                C_b = (
                    self.cost_bbox * cost_bbox
                    + self.cost_class * cost_class
                    - self.cost_giou * giou
                )

                C_b = C_b.float()
                if C_b.numel() > 0:
                    finite_mask = C_b.isfinite()
                    max_cost = C_b.where(finite_mask, C_b.new_tensor(0.0)).max()
                    C_b = C_b.where(finite_mask, max_cost * 2)

                cost_info[layer_idx][b_idx] = (C_b, num_targets_b)

        # Phase 2: Pad and batch-transfer ALL cost matrices with a single .cpu()
        active_entries = []  # (layer_idx, b_idx, num_targets, local_index)
        gpu_matrices = []
        max_targets = 0
        for layer_idx in range(num_layers):
            for b_idx in range(bs):
                info = cost_info[layer_idx][b_idx]
                if info is not None:
                    C_b, nt = info
                    max_targets = max(max_targets, nt)
                    active_entries.append((layer_idx, b_idx, nt, len(gpu_matrices)))
                    gpu_matrices.append(C_b)

        # Build per-layer result
        all_indices = [[None] * bs for _ in range(num_layers)]
        for key, val in all_empty_indices.items():
            all_indices[key[0]][key[1]] = val

        if gpu_matrices:
            padded = torch.stack(
                [F.pad(C, (0, max_targets - C.shape[1]), value=1e9) for C in gpu_matrices]
            )
            padded_cpu = padded.cpu()  # single sync!

            for layer_idx, b_idx, nt, local_idx in active_entries:
                C_b = padded_cpu[local_idx, :, :nt]
                C_groups = C_b.split(g_num_queries, dim=0)
                matched_pred = []
                matched_tgt = []
                for g_i, C_g in enumerate(C_groups):
                    row_ind, col_ind = linear_sum_assignment(C_g.numpy())
                    matched_pred.append(row_ind + g_i * g_num_queries)
                    matched_tgt.append(col_ind)
                all_indices[layer_idx][b_idx] = (
                    torch.as_tensor(np.concatenate(matched_pred), dtype=torch.int64),
                    torch.as_tensor(np.concatenate(matched_tgt), dtype=torch.int64),
                )

        return all_indices


def build_matcher(args):
    if args.segmentation_head:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
        )


def _set_criterion_forward_batched(self, outputs, targets):
    """SetCriterion.forward with all matcher calls batched into one GPU→CPU sync."""
    from rfdetr.util.misc import is_dist_avail_and_initialized, get_world_size

    group_detr = self.group_detr if self.training else 1
    outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

    # Gather all output dicts that need matching
    all_output_dicts = [outputs_without_aux]
    if "aux_outputs" in outputs:
        all_output_dicts.extend(outputs["aux_outputs"])
    if "enc_outputs" in outputs:
        all_output_dicts.append(outputs["enc_outputs"])

    # Single batched matcher call — one GPU→CPU sync for all layers
    all_indices = self.matcher.forward_multi(all_output_dicts, targets, group_detr=group_detr)

    # Compute num_boxes
    num_boxes = sum(len(t["labels"]) for t in targets)
    if not self.sum_group_losses:
        num_boxes = num_boxes * group_detr
    num_boxes = torch.as_tensor(
        [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
    )
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(num_boxes)
    num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

    # Main output losses
    layer_idx = 0
    indices = all_indices[layer_idx]
    losses = {}
    for loss in self.losses:
        losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
    layer_idx += 1

    # Auxiliary losses
    if "aux_outputs" in outputs:
        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            indices = all_indices[layer_idx]
            layer_idx += 1
            for loss in self.losses:
                kwargs = {}
                if loss == "labels":
                    kwargs = {"log": False}
                l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

    # Encoder losses
    if "enc_outputs" in outputs:
        enc_outputs = outputs["enc_outputs"]
        indices = all_indices[layer_idx]
        for loss in self.losses:
            kwargs = {}
            if loss == "labels":
                kwargs["log"] = False
            l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
            l_dict = {k + "_enc": v for k, v in l_dict.items()}
            losses.update(l_dict)

    return losses


# Patch the matcher — need to patch both the module attribute AND lwdetr's
# local binding (which was created by `from rfdetr.models.matcher import build_matcher`).
matcher.HungarianMatcher = HungarianMatcher
matcher.build_matcher = build_matcher
lwdetr.build_matcher = build_matcher

# Patch loss functions — these modify the class directly, so all instances are affected.
lwdetr.SetCriterion.loss_labels = _loss_labels_memeff
lwdetr.SetCriterion.loss_boxes = _loss_boxes_memeff

# Patch SetCriterion.forward to batch all matcher calls into a single GPU→CPU sync.
# NOTE: Disabled — the 5-call pattern has better GPU/CPU overlap than one big batch.
# lwdetr.SetCriterion.forward = _set_criterion_forward_batched

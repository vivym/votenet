from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.layers import huber_loss
from votenet.utils.registry import Registry
from votenet.structures import Instances, Boxes, BoxMode

from .box_head import build_box_head
from ..pooler import ROIGridPooler

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""


def build_roi_heads(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg)


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        pooler: nn.Module,
        box_head: nn.Module,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pooler = pooler
        self.box_head = box_head

    @classmethod
    def from_config(cls, cfg):
        grid_size = cfg.MODEL.ROI_HEADS.GRID_SIZE
        seed_feature_dim = cfg.MODEL.ROI_HEADS.SEED_FEATURE_DIM

        return {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "pooler": ROIGridPooler(grid_size, seed_feature_dim),
            "box_head": build_box_head(cfg),
        }

    def forward(
            self,
            seed_xyz: torch.Tensor,
            seed_features: torch.Tensor,
            voted_features: torch.Tensor,
            proposals: List[Instances],
    ):
        pooled_features = self.pooler(seed_xyz, seed_features, proposals)
        # TODO: more fusion strategy
        features = torch.cat([voted_features, pooled_features], dim=1)

        pred_cls_logits, pred_box_deltas, pred_heading_deltas, pred_centerness = self.box_head(features)

        if self.training:
            losses = self.losses(
                pred_cls_logits, pred_box_deltas, pred_heading_deltas, proposals
            )
            return None, losses
        else:
            losses = {}

        # TODO: yield proposals when training
        if not self.training or False:
            pred_instances = self.predict_instances(
                pred_cls_logits, pred_box_deltas, pred_heading_deltas, proposals
            )
        else:
            pred_instances = None
        return pred_instances, losses

    @torch.no_grad()
    def predict_instances(
            self,
            pred_cls_logits: torch.Tensor,  # (bs, c, num_proposals)
            pred_box_deltas: torch.Tensor,
            pred_heading_deltas: Optional[torch.Tensor],
            proposals: List[Instances],
    ):
        batch_size = pred_cls_logits.size(0)
        num_proposals = pred_cls_logits.size(-1)

        pred_scores, pred_classes = pred_cls_logits.detach().sigmoid().max(dim=1)

        pred_boxes = torch.stack([x.pred_boxes.get_tensor() for x in proposals])
        pred_box_deltas = torch.gather(
            pred_box_deltas.detach(), dim=2, index=pred_classes.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 1, 6)
        ).view(batch_size, num_proposals, 6)
        pred_boxes[..., 3:9] += pred_box_deltas

        if pred_heading_deltas is not None:
            pred_heading_angles = torch.stack([x.pred_heading_angles for x in proposals])
            pred_heading_deltas = torch.gather(
                pred_heading_deltas.detach(), dim=2, index=pred_classes.view(batch_size, num_proposals, 1)
            ).view(batch_size, num_proposals)
            pred_heading_angles += pred_heading_deltas
            pred_boxes = torch.cat([pred_boxes, pred_heading_angles], dim=-1)

        instances = []
        for pred_scores_i, pred_classes_i, pred_boxes_i in zip(
                pred_scores, pred_classes, pred_boxes
        ):
            pred_boxes_i = Boxes.from_tensor(
                pred_boxes_i, mode=BoxMode.XYZLBDRFU_ABS
            ).convert(BoxMode.XYZWDH_ABS)

            # TODO: batch_nms_3d
            # keep = batch_nms_3d(pred_boxes_i, pred_scores_i)

            instances_i = Instances()
            instances_i.pred_classes = pred_classes_i
            instances_i.pred_boxes = pred_boxes_i
            instances_i.scores = pred_scores_i

            instances.append(instances_i)

        return instances

    def losses(
            self,
            pred_cls_logits: torch.Tensor,  # (bs, c, num_proposals)
            pred_box_deltas: torch.Tensor,  # (bs, num_proposals, 6)
            pred_heading_deltas: Optional[torch.Tensor],
            proposals: List[Instances],
    ):
        batch_size = pred_cls_logits.size(0)
        num_proposals = pred_cls_logits.size(-1)
        normalizer = batch_size * num_proposals

        gt_classes = torch.stack([x.gt_classes for x in proposals])
        gt_boxes = torch.stack([x.gt_boxes.get_tensor()[:, 3:9] for x in proposals])
        pred_boxes = torch.stack([x.pred_boxes.get_tensor()[:, 3:9] for x in proposals])
        gt_box_deltas = gt_boxes - pred_boxes

        losses = {}
        losses["loss_cls"] = F.cross_entropy(
            pred_cls_logits,
            gt_classes,
            reduction="sum",
        ) / normalizer

        fg_mask = gt_classes < self.num_classes
        pred_box_deltas = torch.gather(
            pred_box_deltas, dim=2, index=gt_classes.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 1, 6)
        ).view(batch_size, num_proposals, 6)
        # TODO: configurable
        losses["loss_box_reg"] = huber_loss(
            pred_box_deltas[fg_mask, :],
            gt_box_deltas[fg_mask, :],
            beta=0.15,
            reduction="sum",
        ) / normalizer

        if pred_heading_deltas is not  None:
            gt_heading_deltas = torch.stack([x.gt_heading_deltas for x in proposals])

            pred_heading_deltas = torch.gather(
                pred_heading_deltas, dim=2, index=gt_classes.view(batch_size, num_proposals, 1)
            ).view(batch_size, num_proposals)
            losses["loss_angle_reg"] = huber_loss(
                pred_heading_deltas[fg_mask],
                gt_heading_deltas[fg_mask],
                beta=1.0,
                reduction="sum",
            ) / normalizer

        return losses

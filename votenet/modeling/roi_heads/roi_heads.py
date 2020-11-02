from typing import Optional, List

import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.layers import huber_loss
from votenet.utils.registry import Registry
from votenet.structures import Instances

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
        pooler: nn.Module,
        box_head: nn.Module,
    ):
        super().__init__()

        self.pooler = pooler
        self.box_head = box_head

    @classmethod
    def from_config(cls, cfg):
        grid_size = cfg.MODEL.ROI_HEADS.GRID_SIZE
        seed_feature_dim = cfg.MODEL.ROI_HEADS.SEED_FEATURE_DIM

        return {
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
            pred_instances = self.predict_instances(
                pred_cls_logits, pred_box_deltas, pred_heading_deltas, proposals
            )
            return pred_instances, {}

    @torch.no_grad()
    def predict_instances(
            self,
            pred_cls_logits: torch.Tensor,
            pred_box_deltas: torch.Tensor,
            pred_heading_deltas: torch.Tensor,
            proposals: List[Instances],
    ):
        device = pred_cls_logits.device
        batch_size = pred_cls_logits.size(0)

        pred_scores, pred_classes = pred_cls_logits.sigmoid().max(dim=2)

        batch_inds = torch.arange(0, batch_size, dtype=torch.int64, device=device)

        pred_box_reg = torch.stack([x.pred_box_reg for x in proposals])
        pred_box_reg += pred_box_deltas[batch_inds, pred_classes]
        pred_origins = torch.stack([x.pred_origins for x in proposals])
        p1 = pred_origins - pred_box_reg[:, :, :3]
        p2 = pred_origins + pred_box_reg[:, :, 3:]

        pred_heading_angles = torch.stack([x.pred_heading_angles for x in proposals])
        pred_heading_angles += pred_heading_deltas[batch_inds, pred_classes]

        pred_boxes = torch.cat([(p1 + p2) / 2., (p2 - p1), pred_heading_angles], dim=2)

        instances = []
        for pred_scores_i, pred_classes_i, pred_boxes_i in zip(
                pred_scores, pred_classes, pred_boxes
        ):
            instances_i = Instances()
            instances_i.pred_classes = pred_classes_i
            instances_i.pred_scores = pred_scores_i
            instances_i.pred_boxes = pred_boxes_i

            instances.append(instances_i)

        return instances

    def losses(
            self,
            pred_cls_logits: torch.Tensor,
            pred_box_deltas: torch.Tensor,
            pred_heading_deltas: torch.Tensor,
            proposals: List[Instances],
    ):
        device = pred_cls_logits.device
        batch_size = pred_cls_logits.size(0)
        num_proposals = pred_cls_logits.size(1)
        normalizer = batch_size * num_proposals

        gt_classes = torch.stack([x.gt_classes for x in proposals])
        gt_box_reg = torch.stack([x.gt_box_reg for x in proposals])
        gt_heading_deltas = torch.stack([x.gt_heading_deltas for x in proposals])

        losses = {}

        losses["loss_cls"] = F.cross_entropy(
            pred_cls_logits.permute(0, 2, 1),
            gt_classes,
            reduction="sum",
        ) / normalizer

        batch_inds = torch.arange(0, batch_size * num_proposals, dtype=torch.int64, device=device)
        print(gt_box_reg.size(), pred_box_deltas.size(), pred_box_deltas[batch_inds, gt_classes].size())
        # TODO: configurable
        losses["loss_box_reg"] = huber_loss(
            pred_box_deltas[batch_inds, gt_classes],
            gt_box_reg,
            beta=0.15,
            reduction="sum",
        ) / normalizer

        losses["loss_angle_reg"] = huber_loss(
            pred_heading_deltas[batch_inds, gt_classes],
            gt_heading_deltas,
            beta=1.0,
            reduction="sum",
        ) / normalizer

        return losses

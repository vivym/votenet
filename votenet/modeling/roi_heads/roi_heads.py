from typing import Optional, List

import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.utils.registry import Registry
from votenet.structures import Instances

from .box_head import build_box_head

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


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
        box_head = build_box_head(cfg)

        return {
            "box_head": box_head,
        }

    def forward(
            self,
            features: torch.Tensor,
            proposals: List[Instances],
            gt_instances: Optional[List[Instances]] = None,
    ):
        pooled_features = self.pooler(features, proposals)
        # TODO: more fusion strategy
        features = torch.cat([features, pooled_features], dim=1)

        pred_cls_logits, pred_box_deltas, pred_heading_deltas, pred_centerness = self.box_head(features)

        if self.training:
            assert gt_instances is not None
            losses = self.losses(
                pred_cls_logits, pred_box_deltas, pred_heading_deltas, gt_instances
            )
            return proposals, losses
        else:
            pred_instances = None
            return pred_instances, {}

    def losses(
            self,
            pred_cls_logits: torch.Tensor,
            pred_box_deltas: torch.Tensor,
            pred_heading_deltas: torch.Tensor,
            gt_instances: torch.Tensor,
    ):
        batch_size = pred_cls_logits.size(0)
        num_proposals = pred_cls_logits.size(1)
        normalizer = batch_size * num_proposals

        gt_classes = torch.stack([x.gt_classes for x in gt_instances])

        losses = {}

        losses["loss_cls"] = F.binary_cross_entropy_with_logits(

        )
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.layers import huber_loss, batched_nms_3d
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
            pred_cls_logits: torch.Tensor,  # (bs, num_classes, num_proposals)
            pred_box_deltas: torch.Tensor,  # (bs, num_proposals, num_classes, 6)
            pred_heading_deltas: Optional[torch.Tensor],  # (bs, num_proposals, num_classes)
            proposals: List[Instances],
    ):
        # (bs, num_classes, num_proposals) -> (bs, num_proposals, num_classes)
        scores = F.softmax(pred_cls_logits, dim=-2).permute(0, 2, 1)

        instances = []
        for i, (scores_i, pred_box_deltas_i, proposals_i) in enumerate(zip(
                scores, pred_box_deltas.detach(), proposals
        )):
            # (num_proposals, 9)
            proposal_boxes_i = proposals_i.proposal_boxes.get_tensor(BoxMode.XYZLBDRFU_ABS)
            # (num_proposals, num_classes, 6)
            pred_boxes_i = proposal_boxes_i[:, None, 3:9] + pred_box_deltas_i
            # (num_proposals, num_classes, 9)
            pred_boxes_i = torch.cat(
                [
                    proposal_boxes_i[:, None, 0:3].repeat(1, self.num_classes, 1),
                    pred_boxes_i,
                ],
                dim=-1,
            ).contiguous()

            pred_boxes_i = BoxMode.convert(
                pred_boxes_i.view(-1, 9), from_mode=BoxMode.XYZLBDRFU_ABS, to_mode=BoxMode.XYZXYZ_ABS
            ).view(-1, self.num_classes, 6)
            # (num_proposals, num_classes + 1) -> (num_proposals, num_classes)
            scores_i = scores_i[:, :-1]

            # 1. Filter results based on detection scores. It can make NMS more efficient
            #    by filtering out low-confidence detections.
            # TODO: make it configurable
            filter_mask = scores_i > 0.05  # score_thresh  # R x K
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero(as_tuple=False)
            pred_boxes_i = pred_boxes_i[filter_mask]
            scores_i = scores_i[filter_mask]

            # 2. Apply NMS for each class independently.
            # TODO: make it configurable
            keep = batched_nms_3d(pred_boxes_i, scores_i, filter_inds[:, 1], 0.25)
            """
            topk_per_image = 256
            if topk_per_image >= 0:
                keep = keep[:topk_per_image]
            """
            pred_boxes_i, scores_i, filter_inds = pred_boxes_i[keep], scores_i[keep], filter_inds[keep]

            if pred_heading_deltas is not None:
                """
                pred_heading_angles_i = proposals_i.pred_heading_angles
                pred_heading_deltas_i = pred_heading_deltas[i].detach()[fg_mask, fg_pred_classes]
                pred_heading_angles_i += pred_heading_deltas_i
                pred_boxes_i = torch.cat([pred_boxes_i, pred_heading_angles_i], dim=-1)
                """
                raise NotImplementedError

            pred_boxes_i = Boxes.from_tensor(pred_boxes_i, mode=BoxMode.XYZXYZ_ABS)

            instances_i = Instances()
            instances_i.pred_classes = filter_inds[:, 1]
            instances_i.pred_boxes = pred_boxes_i.convert(BoxMode.XYZWDH_ABS)
            instances_i.scores = scores_i
            instances.append(instances_i)

        return instances

    def losses(
            self,
            pred_cls_logits: torch.Tensor,  # (bs, num_classes + 1, num_proposals)
            pred_box_deltas: torch.Tensor,  # (bs, num_proposals, num_classes * 6)
            pred_heading_deltas: Optional[torch.Tensor],  # (bs, num_proposals, num_classes)
            proposals: List[Instances],
    ):
        batch_size = pred_cls_logits.size(0)
        num_proposals = pred_cls_logits.size(-1)
        normalizer = batch_size * num_proposals

        gt_classes = torch.stack([x.gt_classes for x in proposals])
        gt_boxes = torch.stack([x.gt_boxes.get_tensor()[:, 3:9] for x in proposals])
        proposal_boxes = torch.stack([x.proposal_boxes.get_tensor()[:, 3:9] for x in proposals])
        gt_box_deltas = gt_boxes - proposal_boxes

        losses = {}
        losses["loss_cls"] = F.cross_entropy(
            pred_cls_logits,
            gt_classes,
            reduction="mean",
        )

        fg_inds = torch.nonzero(
            (gt_classes >= 0) & (gt_classes < self.num_classes), as_tuple=True
        )
        fg_gt_classes = gt_classes[fg_inds]   # (num_fg,)
        # TODO: configurable
        losses["loss_box_loc"] = huber_loss(
            pred_box_deltas[fg_inds + (fg_gt_classes, )],
            gt_box_deltas[fg_inds],
            beta=0.5,
            reduction="sum",
        ) / normalizer

        if pred_heading_deltas is not None:
            gt_heading_deltas = torch.stack([x.gt_heading_deltas for x in proposals])

            losses["loss_angle_reg"] = huber_loss(
                pred_heading_deltas[fg_inds + (fg_gt_classes, )],
                gt_heading_deltas[fg_inds],
                beta=np.pi / 12.,
                reduction="sum",
            ) / normalizer

        return losses

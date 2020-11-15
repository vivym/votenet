from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.layers import huber_loss, batched_nms_3d, nms_3d, sigmoid_focal_loss
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
            box_reg_loss_weight: float,
            cls_loss_weight: float,
            cls_loss_type: str,
            objectness_loss_type: str,
            cls_agnostic_bbox_reg: bool,
            pooler: nn.Module,
            box_head: nn.Module,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.box_reg_loss_weight = box_reg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.cls_loss_type = cls_loss_type
        self.objectness_loss_type = objectness_loss_type
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.pooler = pooler
        self.box_head = box_head

    @classmethod
    def from_config(cls, cfg):
        grid_size = cfg.MODEL.ROI_HEADS.GRID_SIZE
        seed_feature_dim = cfg.MODEL.ROI_HEADS.SEED_FEATURE_DIM

        return {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "box_reg_loss_weight": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
            "cls_loss_weight": cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT,
            "cls_loss_type": cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_TYPE,
            "objectness_loss_type": cfg.MODEL.ROI_BOX_HEAD.OBJECTNESS_LOSS_TYPE,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
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

        pred_objectness_logits, pred_cls_logits, pred_box_deltas, pred_heading_deltas, pred_centerness = \
            self.box_head(features)

        if self.training:
            losses = self.losses(
                pred_objectness_logits, pred_cls_logits, pred_box_deltas,
                pred_centerness, pred_heading_deltas, proposals
            )
            return None, losses
        else:
            losses = {}

        # TODO: yield proposals when training
        if not self.training or False:
            pred_instances = self.predict_instances(
                pred_objectness_logits, pred_cls_logits, pred_box_deltas, pred_heading_deltas, proposals
            )
        else:
            pred_instances = None
        return pred_instances, losses

    @torch.no_grad()
    def predict_instances(
            self,
            pred_objectness_logits: Optional[torch.Tensor],  # (bs, num_proposals, 2)
            pred_cls_logits: torch.Tensor,  # (bs, num_proposals, num_classes + 1)
            pred_box_deltas: torch.Tensor,  # (bs, num_proposals, num_classes, 6)
            pred_heading_deltas: Optional[torch.Tensor],  # (bs, num_proposals, num_classes)
            proposals: List[Instances],
    ):
        # (bs, 2, num_proposals) -> (bs, num_proposals)
        if pred_objectness_logits is not None:
            # objectness = F.softmax(pred_objectness_logits, dim=-2)[:, 1, :]
            pass

        # (bs, num_proposals, num_classes + 1)
        scores = F.softmax(pred_cls_logits, dim=-1)

        instances = []
        for i, (scores_i, pred_box_deltas_i, proposals_i) in enumerate(zip(
                scores, pred_box_deltas.detach(), proposals
        )):
            # (num_proposals, 6)
            proposal_boxes_i = proposals_i.proposal_boxes.get_tensor(assert_mode=BoxMode.XYZLBDRFU_ABS)
            if self.cls_agnostic_bbox_reg:
                # (num_proposals, 6)
                pred_boxes_i = proposal_boxes_i + pred_box_deltas_i
                # (num_proposals, num_classes, 6)
                pred_origins = proposals_i.proposal_boxes.get("origins")
            else:
                # (num_proposals, num_classes, 6)
                pred_boxes_i = proposal_boxes_i[:, None, :] + pred_box_deltas_i
                # (num_proposals, num_classes, 6)
                pred_origins = proposals_i.proposal_boxes.get("origins")[:, None, :].repeat(1, self.num_classes, 1)

            pred_boxes_i, _ = BoxMode.convert(
                pred_boxes_i, from_mode=BoxMode.XYZLBDRFU_ABS, to_mode=BoxMode.XYZXYZ_ABS,
                origins=pred_origins,
            )
            # (num_proposals, num_classes + 1) -> (num_proposals, num_classes)
            scores_i = scores_i[:, :-1]

            # 1. Filter results based on detection scores. It can make NMS more efficient
            #    by filtering out low-confidence detections.
            # TODO: make it configurable
            filter_mask = scores_i > 0.05  # score_thresh  # R x K
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero(as_tuple=False)
            if self.cls_agnostic_bbox_reg:
                pred_boxes_i = pred_boxes_i[filter_inds[:, 0]]
            else:
                pred_boxes_i = pred_boxes_i[filter_mask]
            scores_i = scores_i[filter_mask]

            # 2. Apply NMS for each class independently.
            # TODO: make it configurable
            if self.cls_agnostic_bbox_reg:
                keep = nms_3d(pred_boxes_i, scores_i, 0.25)
            else:
                keep = batched_nms_3d(pred_boxes_i, scores_i, filter_inds[:, 1], 0.25)
                topk_per_image = 256
                if topk_per_image >= 0:
                    keep = keep[:topk_per_image]

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
            pred_objectness_logits: Optional[torch.Tensor],  # (bs, num_proposals, 2)
            pred_cls_logits: torch.Tensor,  # (bs, num_proposals, num_classes + 1)
            pred_box_deltas: torch.Tensor,  # (bs, num_proposals, num_classes, 6)
            pred_centerness: torch.Tensor, # (bs, num_proposals)
            pred_heading_deltas: Optional[torch.Tensor],  # (bs, num_proposals, num_classes)
            proposals: List[Instances],
    ):
        batch_size = pred_cls_logits.size(0)
        num_proposals = pred_cls_logits.size(1)
        normalizer = batch_size * num_proposals
        device = pred_cls_logits.device
        dtype = pred_cls_logits.dtype

        gt_classes = torch.stack([x.gt_classes for x in proposals])
        gt_boxes = torch.stack([x.gt_boxes.get_tensor(assert_mode=BoxMode.XYZLBDRFU_ABS) for x in proposals])
        proposal_boxes = torch.stack(
            [x.proposal_boxes.get_tensor(assert_mode=BoxMode.XYZLBDRFU_ABS) for x in proposals]
        )
        gt_box_deltas = gt_boxes - proposal_boxes

        losses = {}
        if pred_objectness_logits is not None:
            gt_labels = torch.stack([x.gt_labels for x in proposals])
            valid_mask = gt_labels >= 0
            if self.objectness_loss_type == "cross_entropy":
                losses["loss_box_objectness"] = F.cross_entropy(
                    pred_objectness_logits[valid_mask],
                    gt_labels[valid_mask].long(),
                    weight=torch.as_tensor([0.2, 0.8], dtype=dtype, device=device),
                    reduction="sum",
                ) / normalizer
            else:  # binary_cross_entropy_with_logits
                losses["loss_box_objectness"] = F.binary_cross_entropy_with_logits(
                    pred_objectness_logits.squeeze(-1)[valid_mask],
                    gt_labels[valid_mask].float(),
                    reduction="sum",
                ) / normalizer

        valid_mask = gt_classes >= 0
        if self.cls_loss_type == "cross_entropy":
            losses["loss_cls"] = F.cross_entropy(
                pred_cls_logits[valid_mask],
                gt_classes[valid_mask],
                reduction="sum",
            ) / normalizer * self.cls_loss_weight
        elif self.cls_loss_type == "sigmoid_focal_loss":
            valid_inds = torch.nonzero(valid_mask, as_tuple=True)
            valid_gt_classes = gt_classes[valid_inds]
            gt_classes_one_hot = torch.zeros(
                valid_gt_classes.size(0), self.num_classes + 1, dtype=dtype, device=device
            )
            gt_classes_one_hot_inds = torch.arange(
                valid_gt_classes.size(0), dtype=torch.int64, device=device
            )
            gt_classes_one_hot[gt_classes_one_hot_inds, valid_gt_classes] = 1.
            losses["loss_cls"] = sigmoid_focal_loss(
                pred_cls_logits[valid_mask],
                gt_classes_one_hot,
                alpha=0.25,
                gamma=2.0,
                reduction="sum",
            ) / normalizer * self.cls_loss_weight
        else:
            raise NotImplementedError

        fg_inds = torch.nonzero(
            (gt_classes >= 0) & (gt_classes < self.num_classes), as_tuple=True
        )
        fg_gt_classes = gt_classes[fg_inds]  # (num_fg,)
        fg_gt_box_deltas = gt_box_deltas[fg_inds]
        if self.cls_agnostic_bbox_reg:
            # TODO: configurable
            losses["loss_box_loc"] = huber_loss(
                pred_box_deltas[fg_inds],
                fg_gt_box_deltas,
                beta=0.15,
                reduction="sum",
            ) / normalizer * self.box_reg_loss_weight
            assert pred_centerness is None
        else:
            # TODO: configurable
            losses["loss_box_loc"] = huber_loss(
                pred_box_deltas[fg_inds + (fg_gt_classes, )],
                fg_gt_box_deltas,
                beta=0.15,
                reduction="sum",
            ) / normalizer * self.box_reg_loss_weight

        if pred_centerness is not None:
            gt_centerness = torch.stack([x.gt_centerness for x in proposals])
            losses["loss_box_centerness"] = F.binary_cross_entropy_with_logits(
                pred_centerness[fg_inds],
                gt_centerness[fg_inds],
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

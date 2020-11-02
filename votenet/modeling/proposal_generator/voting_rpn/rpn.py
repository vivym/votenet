from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.layers import ShapeSpec, nn_distance, huber_loss
from votenet.structures import Instances
from votenet.utils.events import get_event_storage

from ..build import PROPOSAL_GENERATOR_REGISTRY
from ..rpn_head import build_rpn_head


@PROPOSAL_GENERATOR_REGISTRY.register()
class VotingRPN(nn.Module):
    """
    Region Proposal Network
    """

    @configurable
    def __init__(
            self,
            *,
            num_classes: int,
            rpn_head: nn.Module,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.rpn_head = rpn_head

    @classmethod
    def from_config(cls, cfg):
        ret = {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "rpn_head": build_rpn_head(cfg),
        }

        return ret

    def forward(
            self,
            voted_xyz: torch.Tensor,
            voted_features: torch.Tensor,
            gt_instances: Optional[List[Instances]] = None,
    ):
        # TODO: pack xyz / features / inds up like Instances

        pred_objectness_logits, pred_box_reg, pred_heading_cls_logits, pred_heading_deltas, pred_centerness = \
            self.rpn_head(voted_features)

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_classes, gt_boxes, gt_box_reg = self.label_and_sample_proposals(voted_xyz, gt_instances)
            if pred_heading_cls_logits is not None:
                gt_heading_classes, gt_heading_deltas = self.compute_gt_angles(gt_boxes)
            else:
                gt_heading_classes, gt_heading_deltas = None, None
            losses = self.loss_instances(
                pred_objectness_logits, gt_labels,
                pred_box_reg, gt_box_reg,
                pred_heading_cls_logits, pred_heading_deltas,
                gt_heading_classes, gt_heading_deltas,
            )
        else:
            gt_classes = None
            gt_boxes = None
            gt_box_reg = None
            gt_heading_deltas = None
            losses = {}

        proposals = self.predict_proposals(
            voted_xyz, pred_objectness_logits, pred_box_reg, pred_heading_cls_logits, pred_heading_deltas,
            gt_classes, gt_boxes, gt_box_reg, gt_heading_deltas
        )
        return proposals, losses

    @torch.no_grad()
    def predict_proposals(
            self,
            proposal_xyz: torch.Tensor,
            pred_objectness_logits: torch.Tensor,
            pred_box_reg: torch.Tensor,
            pred_heading_cls_logits: torch.Tensor,
            pred_heading_deltas: torch.Tensor,
            gt_classes: Optional[torch.Tensor] = None,
            gt_boxes: Optional[torch.Tensor] = None,
            gt_box_reg: Optional[torch.Tensor] = None,
            gt_heading_deltas: Optional[torch.Tensor] = None,
    ):
        pred_objectness = pred_objectness_logits.sigmoid()

        pred_heading_class = torch.argmax(pred_heading_cls_logits, dim=2)  # (bs, num_proposals)
        pred_heading_deltas = torch.gather(
            pred_heading_deltas, dim=2, index=pred_heading_class.unsqueeze(-1)
        ).squeeze(-1)  # (bs, num_proposals)

        print(pred_heading_class.size(), pred_heading_deltas.size())
        pred_heading_angles = pred_heading_class.float() * (2 * np.pi / 12) + pred_heading_deltas
        pred_heading_angles = pred_heading_angles % (2 * np.pi)

        proposals = []
        for i, (pred_objectness_i, pred_origins_i, pred_box_reg_i, pred_heading_angles_i) in enumerate(zip(
                pred_objectness, proposal_xyz, pred_box_reg, pred_heading_angles
        )):
            instances = Instances()
            instances.pred_objectness = pred_objectness_i
            instances.pred_origins = pred_origins_i
            instances.pred_box_reg = pred_box_reg_i
            instances.pred_heading_angles = pred_heading_angles_i

            if gt_classes is not None:
                instances.gt_classes = gt_classes[i]

            if gt_boxes is not None:
                assert gt_box_reg is not None
                instances.gt_boxes = gt_boxes[i]
                instances.gt_box_reg = gt_box_reg[i] - pred_box_reg_i

            if gt_heading_deltas is not None:
                instances.gt_heading_deltas = gt_heading_deltas[i] - pred_heading_angles_i

            proposals.append(instances)

        return proposals

    @torch.no_grad()
    def compute_gt_angles(self, gt_boxes: torch.Tensor):
        gt_angles = gt_boxes[..., 6] % (2 * np.pi)
        angle_per_bin = 2 * np.pi / 12
        shifted_angles = (gt_angles + angle_per_bin / 2) % (2 * np.pi)
        gt_heading_classes = shifted_angles / angle_per_bin
        gt_heading_deltas = shifted_angles - (gt_heading_classes * angle_per_bin + angle_per_bin / 2)
        gt_heading_deltas /= angle_per_bin
        return gt_heading_classes.long(), gt_heading_deltas

    @torch.jit.unused
    def loss_instances(
            self, pred_objectness_logits: torch.Tensor, gt_labels: torch.Tensor,
            pred_box_reg: torch.Tensor, gt_box_reg: torch.Tensor,
            pred_heading_cls_logits: Optional[torch.Tensor] = None,
            pred_heading_deltas: Optional[torch.Tensor] = None,
            gt_heading_classes: Optional[torch.Tensor] = None,
            gt_heading_deltas: Optional[torch.Tensor] = None,
    ):
        batch_size = pred_objectness_logits.size(0)
        num_proposals = pred_objectness_logits.size(-1)
        normalizer = batch_size * num_proposals

        pos_mask = gt_labels == 1
        num_pos = pos_mask.sum().item()
        num_neg = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_proposals", num_pos / batch_size)
        storage.put_scalar("rpn/num_neg_proposals", num_neg / batch_size)

        # TODO: make box_reg_loss_type configurable
        losses = {}

        valid_mask = gt_labels >= 0
        losses["loss_objectness"] = F.binary_cross_entropy_with_logits(
            pred_objectness_logits[valid_mask],
            gt_labels[valid_mask].float(),
            reduction="sum",
        ) / normalizer

        losses["loss_rpn_box_reg"] = huber_loss(
            pred_box_reg[pos_mask],
            gt_box_reg[pos_mask],
            beta=1.0,
            reduction="sum",
        ) / normalizer

        if pred_heading_cls_logits is not None:
            assert pred_heading_deltas is not None
            gt_heading_classes = gt_heading_classes[pos_mask]
            losses["loss_rpn_angle_cls"] = F.cross_entropy(
                pred_heading_cls_logits[pos_mask],
                gt_heading_classes,
                reduction="sum",
            ) / normalizer

            # TODO: optimize this
            batch_inds, proposal_inds = torch.nonzero(pos_mask, as_tuple=True)
            losses["loss_rpn_angle_reg"] = huber_loss(
                pred_heading_deltas[batch_inds, proposal_inds, gt_heading_classes],
                gt_heading_deltas[pos_mask],
                beta=1.0,
                reduction="sum",
            ) / normalizer

        # TODO: loss weight
        return losses

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposal_xyz: torch.Tensor, gt_instances: List[Instances]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = proposal_xyz.device
        num_proposals = proposal_xyz.size(1)

        gt_labels = []
        gt_classes = []
        gt_boxes = []
        gt_box_reg = []
        for (proposal_xyz_i, gt_instances_i) in zip(proposal_xyz, gt_instances):
            gt_classes_i = gt_instances_i.gt_classes
            gt_centers_i = gt_instances_i.gt_boxes.tensor[:, :3]
            gt_sizes_i = gt_instances_i.gt_boxes.tensor[:, 3:6]

            dists, inds, _, _ = nn_distance(proposal_xyz_i, gt_centers_i, dist="euclidean")

            gt_classes_i = gt_classes_i[inds]
            gt_sizes_i = gt_sizes_i[inds, :]
            threshold = torch.mean(gt_sizes_i, dim=1) / 2 * (2 / 3)
            gt_labels_i = torch.zeros(num_proposals, dtype=torch.int64, device=device)
            gt_labels_i[dists < threshold] = 1
            gt_classes_i[dists >= threshold] = self.num_classes  # background

            gt_boxes_i = gt_instances_i.gt_boxes.tensor[inds, :]

            box_half_sizes = gt_boxes_i[:, 3:6] / 2.
            box_centers = gt_boxes_i[:, :3]

            gt_box_reg_i = torch.cat([
                proposal_xyz_i - (box_centers - box_half_sizes),
                (box_centers + box_half_sizes) - proposal_xyz_i,
            ], dim=1)

            gt_labels.append(gt_labels_i)
            gt_classes.append(gt_classes_i)
            gt_boxes.append(gt_boxes_i)
            gt_box_reg.append(gt_box_reg_i)

        gt_labels = torch.stack(gt_labels)
        gt_classes = torch.stack(gt_classes)
        gt_boxes = torch.stack(gt_boxes)
        gt_box_reg = torch.stack(gt_box_reg)

        return gt_labels, gt_classes, gt_boxes, gt_box_reg

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.layers import nn_distance, huber_loss
from votenet.structures import Instances, Boxes, BoxMode
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
            box_reg_loss_weight: float,
            objectness_loss_type: str,
            rpn_head: nn.Module,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.box_reg_loss_weight = box_reg_loss_weight
        self.objectness_loss_type = objectness_loss_type
        self.rpn_head = rpn_head

    @classmethod
    def from_config(cls, cfg):
        ret = {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "box_reg_loss_weight": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT,
            "objectness_loss_type": cfg.MODEL.ROI_BOX_HEAD.OBJECTNESS_LOSS_TYPE,
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
            gt_labels, gt_classes, gt_boxes = self.label_and_sample_proposals(voted_xyz, gt_instances)
            if pred_heading_cls_logits is not None:
                assert all(b.has("angles") for b in gt_boxes)
                gt_heading_classes, gt_heading_deltas = self.compute_gt_angles(gt_boxes)
            else:
                gt_heading_classes, gt_heading_deltas = None, None
            losses = self.losses(
                pred_objectness_logits, gt_labels,
                pred_box_reg, gt_boxes,
                pred_heading_cls_logits, gt_heading_classes,
                pred_heading_deltas, gt_heading_deltas,
            )
        else:
            gt_labels = None
            gt_classes = None
            gt_boxes = None
            gt_heading_classes = None
            gt_heading_deltas = None
            losses = {}

        proposals = self.predict_proposals(
            voted_xyz, pred_objectness_logits, pred_box_reg, pred_heading_cls_logits, pred_heading_deltas,
            gt_labels, gt_classes, gt_boxes, gt_heading_classes, gt_heading_deltas
        )
        return proposals, losses

    @torch.no_grad()
    def predict_proposals(
            self,
            proposal_xyz: torch.Tensor,
            pred_objectness_logits: torch.Tensor,
            pred_box_reg: torch.Tensor,
            pred_heading_cls_logits: Optional[torch.Tensor] = None,
            pred_heading_deltas: Optional[torch.Tensor] = None,
            gt_labels: Optional[torch.Tensor] = None,
            gt_classes: Optional[torch.Tensor] = None,
            gt_boxes: Optional[List["Boxes"]] = None,
            gt_heading_classes: Optional[torch.Tensor] = None,
            gt_heading_deltas: Optional[torch.Tensor] = None,
    ):
        # TODO: cross_entropy
        pred_objectness = pred_objectness_logits.detach().sigmoid()

        if pred_heading_cls_logits is not None:
            if gt_heading_classes is None:
                # TODO: not pick the one, pick by score
                heading_classes = torch.argmax(pred_heading_cls_logits.detach(), dim=2)  # (bs, num_proposals)
            else:
                heading_classes = gt_heading_classes
            pred_heading_deltas = torch.gather(
                pred_heading_deltas.detach(), dim=2, index=heading_classes.unsqueeze(-1)
            ).squeeze(-1)  # (bs, num_proposals)

            pred_heading_angles = heading_classes.float() * (2 * np.pi / 12) + pred_heading_deltas
            pred_heading_angles = pred_heading_angles % (2 * np.pi)
        else:
            pred_heading_angles = None

        proposals = []
        for i, (pred_objectness_i, pred_origins_i, pred_box_reg_i) in enumerate(zip(
                pred_objectness, proposal_xyz.detach(), pred_box_reg.detach()
        )):
            instances = Instances()
            instances.pred_objectness = pred_objectness_i
            instances.proposal_boxes = Boxes.from_tensor(
                pred_box_reg_i,
                mode=BoxMode.XYZLBDRFU_ABS,
                origins=pred_origins_i,
            )
            if pred_heading_angles is not None:
                instances.pred_heading_angles = pred_heading_angles[i]

            if gt_labels is not None:
                instances.gt_labels = gt_labels[i]

            if gt_classes is not None:
                instances.gt_classes = gt_classes[i]

            if gt_boxes is not None:
                instances.gt_boxes = gt_boxes[i]

            if gt_heading_deltas is not None:
                instances.gt_heading_deltas = gt_heading_deltas[i]

            proposals.append(instances)

        return proposals

    @torch.no_grad()
    def compute_gt_angles(self, gt_boxes: List["Boxes"]):
        gt_angles = gt_boxes.get("angles") % (2 * np.pi)
        angle_per_bin = 2 * np.pi / 12
        shifted_angles = (gt_angles + angle_per_bin / 2) % (2 * np.pi)
        gt_heading_classes = shifted_angles / angle_per_bin
        gt_heading_deltas = shifted_angles - (gt_heading_classes * angle_per_bin + angle_per_bin / 2)
        gt_heading_deltas /= angle_per_bin
        return gt_heading_classes.long(), gt_heading_deltas

    @torch.jit.unused
    def losses(
            self,
            pred_objectness_logits: torch.Tensor, gt_labels: torch.Tensor,
            pred_box_reg: torch.Tensor, gt_boxes: List["Boxes"],
            pred_heading_cls_logits: Optional[torch.Tensor],
            gt_heading_classes: Optional[torch.Tensor],
            pred_heading_deltas: Optional[torch.Tensor],
            gt_heading_deltas: Optional[torch.Tensor],
    ):
        batch_size = pred_box_reg.size(0)
        num_proposals = pred_box_reg.size(1)
        normalizer = batch_size * num_proposals
        device = pred_box_reg.device
        dtype = pred_box_reg.dtype

        pos_mask = gt_labels == 1
        num_pos = pos_mask.sum().item()
        num_neg = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/pos_ratio", num_pos / normalizer)
        storage.put_scalar("rpn/neg_ratio", num_neg / normalizer)

        # TODO: make box_reg_loss_type configurable
        losses = {}

        valid_mask = gt_labels >= 0
        if self.objectness_loss_type == "binary_cross_entropy_with_logits":
            losses["loss_objectness"] = F.binary_cross_entropy_with_logits(
                pred_objectness_logits[valid_mask],
                gt_labels[valid_mask].float(),
                reduction="sum",
            ) / normalizer
        else:  # cross_entropy
            losses["loss_objectness"] = F.cross_entropy(
                pred_objectness_logits[valid_mask],
                gt_labels[valid_mask].long(),
                weight=torch.as_tensor([0.2, 0.8], dtype=dtype, device=device),
                reduction="sum",
            ) / normalizer

        gt_box_reg = torch.stack([x.get_tensor(assert_mode=BoxMode.XYZLBDRFU_ABS) for x in gt_boxes])
        losses["loss_rpn_loc"] = huber_loss(
            pred_box_reg[pos_mask],
            gt_box_reg[pos_mask],
            beta=0.15,
            reduction="sum",
        ) / normalizer * self.box_reg_loss_weight

        if pred_heading_cls_logits is not None:
            assert pred_heading_deltas is not None
            gt_heading_classes = gt_heading_classes[pos_mask]
            losses["loss_rpn_angle_cls"] = F.cross_entropy(
                pred_heading_cls_logits[pos_mask],
                gt_heading_classes,
                reduction="sum",
            ) / normalizer

            batch_inds, proposal_inds = torch.nonzero(pos_mask, as_tuple=True)
            losses["loss_rpn_angle_delta"] = huber_loss(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, List["Boxes"]]:
        device = proposal_xyz.device
        num_proposals = proposal_xyz.size(1)

        gt_labels = []
        gt_classes = []
        gt_boxes = []
        for (proposal_xyz_i, gt_instances_i) in zip(proposal_xyz, gt_instances):
            gt_classes_i = gt_instances_i.gt_classes
            gt_centers_i = gt_instances_i.gt_boxes.get_centers()
            gt_sizes_i = gt_instances_i.gt_boxes.get_sizes()

            dists, inds, _, _ = nn_distance(proposal_xyz_i, gt_centers_i, dist="euclidean")

            gt_classes_i = gt_classes_i[inds]
            gt_sizes_i = gt_sizes_i[inds, :]
            threshold = torch.sum(gt_sizes_i, dim=-1) / 6 * (2 / 3)
            threshold = threshold ** 2
            gt_labels_i = torch.zeros(num_proposals, dtype=torch.int64, device=device)
            gt_labels_i[dists < threshold] = 1
            # TODO: gt_labels_i[dists < threshold2] = -1
            gt_classes_i[dists >= threshold] = self.num_classes  # background

            gt_boxes_i = gt_instances_i.gt_boxes[inds, :]
            gt_boxes_i = gt_boxes_i.convert(BoxMode.XYZLBDRFU_ABS, origins=proposal_xyz_i)

            gt_labels.append(gt_labels_i)
            gt_classes.append(gt_classes_i)
            gt_boxes.append(gt_boxes_i)

        gt_labels = torch.stack(gt_labels)
        gt_classes = torch.stack(gt_classes)

        return gt_labels, gt_classes, gt_boxes

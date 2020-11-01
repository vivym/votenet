from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from votenet.config import configurable
from votenet.layers import ShapeSpec, nn_distance
from votenet.structures import Instances
from detectron2.utils.events import get_event_storage

from votenet.layers import huber_loss
from votenet.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY


@PROPOSAL_GENERATOR_REGISTRY.register()
class VotingRPN(nn.Module):
    """
    Region Proposal Network
    """

    @configurable
    def __init__(
            self,
            *,
            in_features: List[str],
            vote_net: nn.Module,
            vote_agg_net: nn.Module,
            rpn_head: nn.Module,
    ):
        super().__init__()

        self.in_features = in_features
        self.vote_net = vote_net
        self.vote_agg_net = vote_agg_net
        self.rpn_head = rpn_head

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.VOTING_RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
        }

        return ret

    def forward(
            self,
            seed_xyz: torch.Tensor,
            seed_features: torch.Tensor,
            seed_inds: torch.Tensor,
            gt_instances: Optional[List[Instances]] = None,
            gt_votes: Optional[torch.Tensor] = None,
            gt_votes_mask: Optional[torch.Tensor] = None,
    ):
        # TODO: pack xyz / features / inds up like Instances
        voted_xyz, voted_features = self.vote_net(seed_xyz, seed_features)
        features_norm = torch.norm(voted_features, p=2, dim=1)
        voted_features = voted_features.div(features_norm.unsqueeze(1))

        proposal_xyz, proposal_features = self.vote_agg_net(voted_xyz, voted_features)

        pred_objectness_logits, pred_box_deltas, pred_heading_cls_logits, pred_heading_deltas, pred_centerness = \
            self.rpn_head(proposal_xyz, proposal_features)

        if self.training:
            assert gt_instances is not None and gt_votes is not None and gt_votes_mask,\
                "RPN requires gt_instances, gt_votes and gt_votes_mask in training!"
            gt_labels, gt_boxes = self.label_and_sample_proposals(proposal_xyz, gt_instances)
            losses = {
                "loss_vote": self.loss_vote(voted_xyz, seed_xyz, seed_inds, gt_votes, gt_votes_mask),
            }
            losses.update(self.loss_instances(pred_objectness_logits, gt_labels))
        else:
            gt_boxes = None
            losses = {}

        proposals = self.predict_proposals(
            proposal_xyz, pred_objectness_logits, pred_box_deltas,
            pred_heading_cls_logits, pred_heading_deltas, gt_boxes,
        )
        return proposals, losses

    def predict_proposals(
            self,
            proposal_xyz: torch.Tensor,
            pred_objectness_logits: torch.Tensor,
            pred_box_deltas: torch.Tensor,
            pred_heading_cls_logits: torch.Tensor,
            pred_heading_deltas: torch.Tensor,
            gt_boxes: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            pred_objectness = pred_objectness_logits.sigmoid()

            pred_heading_class = torch.argmax(pred_heading_cls_logits, dim=2)  # (bs, num_proposals)
            pred_heading_class = torch.gather(
                pred_heading_deltas, dim=2, index=pred_heading_class.unsqueeze(-1)
            ).squeeze(-1)  # (bs, num_proposals)

            pred_heading_angles = pred_heading_class.float() * (2 * np.pi / 12) + pred_heading_deltas
            pred_heading_angles = pred_heading_angles % (2 * np.pi)

            proposals = []
            for i, (pred_objectness_i, pred_origins_i, pred_box_deltas_i, pred_heading_angles_i) in enumerate(zip(
                    pred_objectness, proposal_xyz, pred_box_deltas, pred_heading_angles
            )):
                instances = Instances()
                instances.pred_objectness = pred_objectness_i
                instances.pred_origins = pred_origins_i
                instances.pred_box_deltas = pred_box_deltas_i
                instances.pred_heading_angles = pred_heading_angles_i

                if gt_boxes is not None:
                    instances.gt_boxes = gt_boxes[i]

                proposals.append(instances)

            return proposals

    @torch.jit.unused
    def loss_instances(
            self, pred_objectness_logits: torch.Tensor, gt_labels: torch.Tensor,
            pred_heading_cls_logits: torch.Tensor, pred_heading_deltas: torch.Tensor,
            gt_boxes: torch.Tensor,
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

        if pred_heading_cls_logits is not None:
            assert pred_heading_deltas is not None
            with torch.no_grad():
                gt_angles = gt_boxes[: 6] % (2 * np.pi)
                angle_per_bin = 2 * np.pi / 12
                shifted_angles = (gt_angles + angle_per_bin / 2) % (2 * np.pi)
                gt_heading_classes = shifted_angles / angle_per_bin
                gt_heading_deltas = shifted_angles - (gt_heading_classes * angle_per_bin + angle_per_bin / 2)
                gt_heading_deltas /= angle_per_bin

            gt_heading_classes = gt_heading_classes[pos_mask]
            losses["loss_rpn_angle_cls"] = F.cross_entropy(
                pred_heading_cls_logits[pos_mask],
                gt_heading_classes,
                reduction="sum",
            ) / normalizer

            losses["loss_rpn_angle_reg"] = huber_loss(
                pred_heading_deltas[torch.nonzero(pos_mask, as_tuple=True), gt_heading_classes],
                gt_heading_deltas[pos_mask],
                beta=1.0,
                reduction="sum",
            ) / normalizer

        # TODO: loss weight
        return losses

    @torch.jit.unused
    def loss_vote(
            self,
            voted_xyz: torch.Tensor, seed_xyz: torch.Tensor, seed_inds: torch.Tensor,
            gt_votes: torch.Tensor, gt_votes_mask: torch.Tensor,
    ):
        batch_size = voted_xyz.size(0)
        num_seeds = seed_xyz.size(1)

        gt_votes_mask = torch.gather(gt_votes_mask, dim=1, index=seed_inds)
        seed_inds = seed_inds.unsqueeze(-1).expand(-1, -1, 9)
        gt_votes = torch.gather(gt_votes, dim=1, index=seed_inds)
        gt_votes += seed_xyz.repeat(1, 1, 3)

        # bs, num_seeds, vote_factor*3
        voted_xyz = voted_xyz.view(batch_size * num_seeds, -1, 3)
        gt_votes = gt_votes.view(batch_size * num_seeds, 3, 3)

        _, _, dist, _ = nn_distance(voted_xyz, gt_votes, dist="l1")
        # (bs * num_seeds, vote_factor) -> (bs, num_seeds)
        vote_dist = torch.min(dist, dim=1).view(batch_size, num_seeds)
        normalizer = gt_votes_mask.sum()
        return torch.sum(vote_dist * gt_votes_mask.float()) / (normalizer + 1e-6)

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposal_xyz: torch.Tensor, gt_instances: List[Instances]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = proposal_xyz.device
        num_proposals = proposal_xyz.size(1)

        gt_labels = []
        gt_boxes = []
        for (proposal_xyz_i, gt_instances_i) in zip(proposal_xyz, gt_instances):
            gt_centers_i = gt_instances_i.gt_boxes.tensor[:, :3]
            gt_sizes_i = gt_instances_i.gt_boxes.tensor[:, 3:6]

            dists, inds, _, _ = nn_distance(proposal_xyz_i, gt_centers_i, dist="euclidean")

            threshold = torch.mean(gt_sizes_i, dim=1) / 2 * (2 / 3)
            gt_labels_i = torch.zeros(num_proposals, dtype=torch.int64, device=device)
            gt_labels_i[dists < threshold] = 1

            gt_boxes_i = gt_instances_i.gt_boxes.tensor[inds, :]

            gt_labels.append(gt_labels_i)
            gt_boxes.append(gt_boxes_i)

        gt_labels = torch.stack(gt_labels)
        gt_boxes = torch.stack(gt_boxes)

        return gt_labels, gt_boxes

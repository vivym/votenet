from typing import List

import torch
from torch import nn

from votenet.config import configurable
from votenet.structures import Instances

from ..backbone import Backbone, build_backbone
from ..vote_generator import build_vote_generator
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class GeneralizedVoteNet(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            vote_generator: nn.Module,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.vote_generator = vote_generator
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

    @classmethod
    def from_config(cls, cfg):
        return {
            "backbone": build_backbone(cfg),
            "vote_generator": build_vote_generator(cfg),
            "proposal_generator": build_proposal_generator(cfg),
            "roi_heads": build_roi_heads(cfg),
        }

    @property
    def device(self):
        param = next(self.parameters())
        return param.device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        points = self.preprocess_points(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_point_votes = [x["point_votes"].to(self.device) for x in batched_inputs]
            gt_point_votes_mask = [x["point_votes_mask"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            gt_point_votes = None
            gt_point_votes_mask = None

        features = self.backbone(points)
        # TODO: pack up
        seed_xyz = features["fp2"]["xyz"]
        seed_features = features["fp2"]["features"]
        seed_inds = features["fp2"]["inds"].long()

        voted_xyz, voted_features, voted_inds, vote_losses = self.vote_generator(
            seed_xyz, seed_features, seed_inds, gt_point_votes, gt_point_votes_mask
        )

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                voted_xyz, voted_features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(seed_xyz, seed_features, voted_features, proposals)

        losses = {}
        losses.update(vote_losses)
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training

        points = self.preprocess_points(batched_inputs)

        features = self.backbone(points)
        seed_xyz = features["fp2"]["xyz"]
        seed_features = features["fp2"]["features"]
        seed_inds = features["fp2"]["inds"]

        voted_xyz, voted_features, voted_inds, vote_losses = self.vote_generator(
            seed_xyz, seed_features, seed_inds
        )

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(
                voted_xyz, voted_features
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        instances, _ = self.roi_heads(seed_xyz, seed_features, voted_features, proposals)

        return GeneralizedVoteNet._postprocess(instances)

    def preprocess_points(self, batched_inputs):
        points = [x["points"].to(self.device) for x in batched_inputs]
        return torch.stack(points)

    @staticmethod
    def _postprocess(instances):
        results = []
        for instances_i in instances:
            # TODO: do postprocessing
            results.append({
                "instances": instances_i,
            })
        return results

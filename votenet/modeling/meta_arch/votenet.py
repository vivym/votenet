from typing import Optional

import torch
from torch import nn

from votenet.config import configurable

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
            proposal_generator: Optional[nn.Module],
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

        """
        state_dict = torch.load(
            "../_votenet/logs/scannet/cps/checkpoint_epoch_0.tar",
            map_location="cpu",
        )["model_state_dict"]
        points = torch.load("../_votenet/points.pth", map_location="cpu").cuda()

        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone_net."):
                k = k.replace("backbone_net.", "")
                backbone_state_dict[k] = v.cuda()
        self.backbone.load_state_dict(backbone_state_dict)
        """

        features = self.backbone(points)
        # TODO: pack up
        seed_xyz = features["fp2"]["xyz"]
        seed_features = features["fp2"]["features"]
        seed_inds = features["fp2"]["inds"].long()

        """
        vote_generator_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vgen."):
                k = k.replace("vgen.", "vote_net.")
                vote_generator_state_dict[k] = v.cuda()
            if k.startswith("v_agg."):
                k = k.replace("v_agg.", "vote_agg_net.")
                vote_generator_state_dict[k] = v.cuda()
        self.vote_generator.load_state_dict(vote_generator_state_dict)
        """

        voted_xyz, voted_features, voted_inds, vote_losses = self.vote_generator(
            seed_xyz, seed_features, seed_inds, gt_point_votes, gt_point_votes_mask
        )

        if self.proposal_generator is not None:
            """
            proposal_generator_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("box_proposal_0."):
                    if "seed_aggregation" in k:
                        continue
                    if "reduce_dim" in k:
                        continue
                    k = k.replace("box_proposal_0.", "rpn_head.")
                    k = k.replace("conv1", "convs.0")
                    k = k.replace("bn1", "convs.1")
                    k = k.replace("conv2", "convs.3")
                    k = k.replace("bn2", "convs.4")
                    k = k.replace("conv3", "predictor")
                    proposal_generator_state_dict[k] = v.cuda()
            self.proposal_generator.load_state_dict(proposal_generator_state_dict)
            """

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

        voted_xyz, voted_features, voted_inds, _ = self.vote_generator(
            seed_xyz, seed_features, seed_inds
        )

        if self.proposal_generator is not None:
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

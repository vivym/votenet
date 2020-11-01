import torch
from torch import nn

from votenet.config import configurable

from votenet.modeling.backbone import Backbone, build_backbone
from votenet.modeling.meta_arch.build import META_ARCH_REGISTRY
from .voting_module import build_voting_module
from .vote_aggregation_module import build_vote_agg_module
from .box_proposal_module_0 import build_box_proposal_module_0


@META_ARCH_REGISTRY.register()
class VoteNet(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            vgen: nn.Module,
            v_agg: nn.Module,
            box_proposal_0: nn.Module,
            box_proposal_1: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.vgen = vgen
        self.v_agg = v_agg
        self.box_proposal_0 = box_proposal_0
        self.box_proposal_1 = box_proposal_1
        self.fusion_type = "concat"  # TODO: configurable

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        vgen = build_voting_module(cfg)
        v_agg = build_vote_agg_module(cfg)
        box_proposal_0 = build_box_proposal_module_0(cfg)

        return {
            "backbone": backbone,
            "vgen": vgen,
            "v_agg": v_agg,
            "box_proposal_0": box_proposal_0,
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

        results = self.backbone(points)

        xyz = results["fp2"]["xyz"]
        features = results["fp2"]["features"]
        seed_xyz = xyz
        seed_features = features
        seed_inds = results["fp2"]["inds"]

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        vote_xyz = xyz
        vote_features = features

        aggregated_vote_xyz, aggregated_vote_features, _ = self.v_agg(xyz, features)

        proposal_0 = self.box_proposal_0(
            aggregated_vote_xyz, aggregated_vote_features,
            seed_xyz, seed_features,
        )

        if self.fusion_type == "concat":
            new_vote_features = torch.cat(
                (aggregated_vote_features, proposal_0["new_seed_features"]), dim=1
            )  # (B, 256, num_proposal)
        elif self.fusion_type == "addition":
            new_vote_features = aggregated_vote_features + proposal_0["new_seed_features"]  # (B, 128, num_proposal)
        else:
            raise NotImplementedError

        results = self.box_proposal_1(new_vote_features, proposal_0)

        return results

    def inference(self, batched_inputs):
        assert not self.training

        points = self.preprocess_points(batched_inputs)

        results = self.backbone(points)

    def preprocess_points(self, batched_inputs):
        points = [x["points"].to(self.device) for x in batched_inputs]
        return torch.stack(points)

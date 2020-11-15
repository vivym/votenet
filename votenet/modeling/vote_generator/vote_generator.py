from typing import Optional, List

import torch
from torch import nn

from votenet.config import configurable
from votenet.layers import nn_distance

from .build import VOTE_GENERATOR_REGISTRY
from .vote_net import build_vote_net
from .vote_agg_net import build_vote_agg_net


@VOTE_GENERATOR_REGISTRY.register()
class StandardVoteGenerator(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            vote_net: nn.Module,
            vote_agg_net: nn.Module,
    ):
        super().__init__()

        self.vote_net = vote_net
        self.vote_agg_net = vote_agg_net

    @classmethod
    def from_config(cls, cfg):
        ret = {
            "vote_net": build_vote_net(cfg),
            "vote_agg_net": build_vote_agg_net(cfg),
        }

        return ret

    def forward(
            self,
            seed_xyz: torch.Tensor,
            seed_features: torch.Tensor,
            seed_inds: torch.Tensor,
            gt_votes: Optional[List[torch.Tensor]] = None,
            gt_votes_mask: Optional[List[torch.Tensor]] = None,
    ):
        voted_xyz, voted_features = self.vote_net(seed_xyz, seed_features)
        features_norm = torch.norm(voted_features, p=2, dim=1)
        voted_features = voted_features.div(features_norm.unsqueeze(1))

        if self.training:
            assert gt_votes is not None and gt_votes_mask, \
                "RPN requires gt_votes and gt_votes_mask in training!"
            losses = self.losses(voted_xyz, seed_xyz, seed_inds, gt_votes, gt_votes_mask)
        else:
            losses = {}

        voted_xyz, voted_features, voted_inds = self.vote_agg_net(voted_xyz, voted_features, seed_xyz)

        return voted_xyz, voted_features, voted_inds, losses

    @torch.jit.unused
    def losses(
            self,
            voted_xyz: torch.Tensor, seed_xyz: torch.Tensor, seed_inds: torch.Tensor,
            gt_votes: List[torch.Tensor], gt_votes_mask: List[torch.Tensor],
    ):
        batch_size = voted_xyz.size(0)
        num_seeds = seed_xyz.size(1)

        gt_votes_list, gt_votes_mask_list = [], []
        for seed_inds_i, gt_votes_i, gt_votes_mask_i in zip(seed_inds, gt_votes, gt_votes_mask):
            gt_votes_list.append(gt_votes_i[seed_inds_i, :])
            gt_votes_mask_list.append(gt_votes_mask_i[seed_inds_i])
        gt_votes = seed_xyz.repeat(1, 1, 3) + torch.stack(gt_votes_list)
        gt_votes_mask = torch.stack(gt_votes_mask_list)

        # bs, num_seeds, vote_factor*3
        voted_xyz = voted_xyz.view(batch_size * num_seeds, -1, 3)
        gt_votes = gt_votes.view(batch_size * num_seeds, 3, 3)

        _, _, dist, _ = nn_distance(voted_xyz, gt_votes, dist="l1")
        # (bs * num_seeds, vote_factor) -> (bs, num_seeds)
        vote_dist, _ = torch.min(dist, dim=-1)

        losses = {
            "loss_vote": torch.mean(vote_dist.view(batch_size, num_seeds)[gt_votes_mask]),
        }

        return losses

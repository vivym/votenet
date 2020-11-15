import torch
import torch.nn as nn

from votenet.layers import furthest_point_sample
from votenet.modeling.backbone import PointnetSAModuleVotes


class VoteAggregationModule(nn.Module):
    def __init__(self, num_proposals, sampling, seed_feat_dim=256):
        super().__init__()

        self.num_proposals = num_proposals
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering/aggregation
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposals,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

    def forward(self, xyz, features, seed_xyz):
        if self.sampling == "vote_fps":
            sample_inds = None
        elif self.sampling == "seed_fps":
            sample_inds = furthest_point_sample(seed_xyz, self.num_proposals)
        elif self.sampling == "random":
            batch_size = seed_xyz.shape[0]
            num_seed = seed_xyz.shape[1]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposals), dtype=torch.int).cuda()
        else:
            raise NotImplementedError

        xyz, features, inds = self.vote_aggregation(xyz, features, sample_inds)

        return xyz, features, inds


def build_vote_agg_net(cfg):
    num_proposals = cfg.MODEL.VOTE_GENERATOR.NUM_PROPOSALS
    sampling_strategy = cfg.MODEL.VOTE_GENERATOR.SAMPLING_STRATEGY
    seed_feature_dim = cfg.MODEL.VOTE_GENERATOR.SEED_FEATURE_DIM

    return VoteAggregationModule(num_proposals, sampling_strategy, seed_feature_dim)

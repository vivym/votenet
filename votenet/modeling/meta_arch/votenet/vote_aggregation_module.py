import torch
import torch.nn as nn

from votenet.layers import furthest_point_sample
from votenet.modeling.backbone import PointnetSAModuleVotes


class VoteAggregationModule(nn.Module):
    def __init__(self, num_proposal, sampling, seed_feat_dim=256):
        super().__init__()

        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering/aggregation
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

    def forward(self, xyz, features, seed_xyz):
        if self.sampling == 'vote_fps':
            sample_inds = None
        elif self.sampling == 'seed_fps':
            sample_inds = furthest_point_sample(seed_xyz, self.num_proposal)
        elif self.sampling == 'random':
            batch_size = seed_xyz.shape[0]
            num_seed = seed_xyz.shape[1]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
        else:
            raise NotImplementedError

        xyz, features, sample_inds = self.vote_aggregation(xyz, features, sample_inds)

        """
        end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,)
        end_points['aggregated_vote_features'] = features  # (batch_size, 128, num_propsoal)
        """

        return xyz, features, sample_inds


def build_vote_agg_module(cfg):
    num_proposals = cfg.MODEL.VOTE_AGG.NUM_PROPOSALS
    sampling_strategy = cfg.MODEL.VOTE_AGG.SAMPLING_STRATEGY
    seed_feature_dim = cfg.MODEL.VOTING_MODULE.SEED_FEATURE_DIM

    return VoteAggregationModule(num_proposals, sampling_strategy, seed_feature_dim)

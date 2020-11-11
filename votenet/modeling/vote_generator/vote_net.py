import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F


class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

        for layer in [self.conv1, self.conv2]:
            weight_init.c2_msra_fill(layer)

        nn.init.normal_(self.conv3.weight, std=0.001)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0)

    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seeds = seed_xyz.shape[1]
        num_votes = num_seeds * self.vote_factor
        x = F.relu(self.bn1(self.conv1(seed_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # (batch_size, (3+out_dim)*vote_factor, num_seed)

        x = x.transpose(2, 1).view(batch_size, num_seeds, self.vote_factor, 3 + self.out_dim)
        offset = x[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.reshape(batch_size, num_votes, 3)

        residual_features = x[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.reshape(batch_size, num_votes, self.out_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()

        return vote_xyz, vote_features


def build_vote_net(cfg):
    vote_factor = cfg.MODEL.VOTE_GENERATOR.VOTE_FACTOR
    seed_feature_dim = cfg.MODEL.VOTE_GENERATOR.SEED_FEATURE_DIM

    return VotingModule(vote_factor, seed_feature_dim)

from typing import Optional, Dict

import torch
from torch import nn
from torch.nn import functional as F

from votenet.modeling.backbone import Backbone, BACKBONE_REGISTRY

from .pointnet2 import PointnetSAModuleVotes, PointnetFPModule

__all__ = ["PointNet2"]


class PointNet2(Backbone):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.
    """

    def __init__(self, input_feature_dim):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, points: torch.Tensor, **kwargs):
        r"""
            Forward pass of the network

            Parameters
            ----------
            points: torch.Tensor
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        xyz, features = self._break_up_pc(points)

        outputs = {}

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(
            xyz, features, inds=kwargs["sa1_inds"] if "sa1_inds" in kwargs else None
        )
        outputs["sa1"] = {
            "xyz": xyz,
            "features": features,
            "inds": fps_inds,
        }

        xyz, features, fps_inds = self.sa2(
            xyz, features, inds=kwargs["sa2_inds"] if "sa2_inds" in kwargs else None
        )
        outputs["sa2"] = {
            "xyz": xyz,
            "features": features,
            "inds": fps_inds,
        }

        xyz, features, fps_inds = self.sa3(
            xyz, features, inds=kwargs["sa3_inds"] if "sa3_inds" in kwargs else None
        )
        outputs["sa3"] = {
            "xyz": xyz,
            "features": features,
            "inds": fps_inds,
        }

        xyz, features, fps_inds = self.sa4(
            xyz, features, inds=kwargs["sa4_inds"] if "sa4_inds" in kwargs else None
        )
        outputs["sa4"] = {
            "xyz": xyz,
            "features": features,
            "inds": fps_inds,
        }

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(
            outputs["sa3"]["xyz"], outputs["sa4"]["xyz"],
            outputs["sa3"]["features"], outputs["sa4"]["features"],
        )
        features = self.fp2(
            outputs["sa2"]["xyz"], outputs["sa3"]["xyz"],
            outputs["sa2"]["features"], features,
        )
        num_seeds = outputs["sa2"]["xyz"].size(1)
        outputs["fp2"] = {
            "xyz": outputs["sa2"]["xyz"],
            "features": features,
            "inds": outputs["sa1"]["inds"][:, :num_seeds]
        }

        return outputs


@BACKBONE_REGISTRY.register()
def build_pointnet2_backbone(cfg):
    use_color = cfg.INPUT.USE_COLOR
    use_height = cfg.INPUT.USE_HEIGHT

    return PointNet2(int(use_color) * 3 + int(use_height) * 1)


class MultiBackbone(Backbone):
    def __init__(self, input_feature_dim):
        super().__init__()
        self.backbones = nn.ModuleList((
            PointNet2(input_feature_dim),
            PointNet2(input_feature_dim),
            PointNet2(input_feature_dim),
            PointNet2(input_feature_dim)
        ))

        self.conv_agg1 = torch.nn.Conv1d(256 * 4, 256 * 2, 1)
        self.bn_agg1 = torch.nn.BatchNorm1d(256 * 2)
        self.conv_agg2 = torch.nn.Conv1d(256 * 2, 256, 1)
        self.bn_agg2 = torch.nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor):
        features0 = self.backbones[0](x)
        sa1_inds = features0["sa1"]["inds"]
        sa2_inds = features0["sa2"]["inds"]
        sa3_inds = features0["sa3"]["inds"]
        sa4_inds = features0["sa4"]["inds"]
        features = [features0]
        for backbone in self.backbones[1:]:
            features_i = backbone(
                x, sa1_inds=sa1_inds, sa2_inds=sa2_inds, sa3_inds=sa3_inds, sa4_inds=sa4_inds
            )
            features.append(features_i)

        xyz = features0["fp2"]["xyz"]
        inds = features0["fp2"]["inds"].long()

        features = torch.cat((
            features[0]["fp2"]["features"],
            features[1]["fp2"]["features"],
            features[2]["fp2"]["features"],
            features[3]["fp2"]["features"],
        ), dim=1)
        features = F.relu(self.bn_agg1(self.conv_agg1(features)))
        features = F.relu(self.bn_agg2(self.conv_agg2(features)))

        return {
            "fp2": {
                "xyz": xyz,
                "features": features,
                "inds": inds
            }
        }


@BACKBONE_REGISTRY.register()
def build_multi_pointnet2_backbone(cfg):
    use_color = cfg.INPUT.USE_COLOR
    use_height = cfg.INPUT.USE_HEIGHT

    return MultiBackbone(int(use_color) * 3 + int(use_height) * 1)

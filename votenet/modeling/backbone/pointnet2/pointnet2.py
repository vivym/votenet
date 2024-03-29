from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from votenet.layers import (
    gather_operation,
    furthest_point_sample,
    QueryAndGroup,
    GroupAll,
    three_nn,
    three_interpolate,
)

from .misc import SharedMLP


class _PointnetSAModuleBase(nn.Module):
    def __init__(self, num_points: int, groupers: nn.ModuleList, mlps: nn.ModuleList):
        super().__init__()

        self.num_points = num_points
        self.groupers = groupers
        self.mlps = mlps

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = gather_operation(
            xyz_flipped,
            furthest_point_sample(xyz, self.num_points)
        ).transpose(1, 2).contiguous() if self.num_points is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, num_points, nsample)

            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], num_points, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], num_points, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], num_points)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    num_points : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            num_points: int,
            radii: List[float],
            num_samples: List[int],
            mlp_specs: List[List[int]],
            use_bn: bool = True,
            use_xyz: bool = True,
            sample_uniformly: bool = False
    ):
        assert len(radii) == len(num_samples) == len(mlp_specs)

        groupers = nn.ModuleList()
        mlps = nn.ModuleList()
        for (radius, num_sample, mlp_spec) in zip(radii, num_samples, mlp_specs):
            self.groupers.append(
                QueryAndGroup(radius, num_sample, use_xyz=use_xyz, sample_uniformly=sample_uniformly)
                if num_points is not None else GroupAll(use_xyz)
            )
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(SharedMLP(mlp_spec, bn=use_bn))

        super().__init__(num_points, groupers, mlps)


class PointnetSAModuleVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = furthest_point_sample(xyz, self.npoint)
        else:
            assert(inds.shape[1] == self.npoint)
        new_xyz = gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        new_features = self.mlp_module(
            grouped_features
        )  # (B, mlp[-1], npoint, nsample)
        if self.pooling == 'max':
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'rbf':
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1,keepdim=False) / (self.sigma**2) / 2) # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(self.nsample) # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


class PointnetFPModule(nn.Module):
    r"""Propagate the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm or not
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *known_feats.size()[0:2], unknown.size(1)
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  #(B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class PointnetSAMoudleAgg(nn.Module):
    def __init__(
            self,
            mlp: List[int],
            radius: float,
            nsample: int,
            use_xyz: bool = True,
            normalize_xyz: bool = True,
            bn: bool = True,
            pooling: str = 'max',
    ):
        super().__init__()
        self.radius = radius
        self.pooling = pooling
        self.grouper = QueryAndGroup(
            radius, nsample, use_xyz=use_xyz, normalize_xyz=normalize_xyz, ret_grouped_xyz=True
        )
        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz, new_xyz, features):
        # (B, C, npoint, nsample)
        # (B, 3, npoint, nsample)
        grouped_features, grouped_xyz = self.grouper(
            xyz, new_xyz, features
        )

        # (B, mlp[-1], npoint, nsample)
        new_features = self.mlp_module(
            grouped_features
        )

        if self.pooling == 'max':
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        else:
            raise NotImplementedError

        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features

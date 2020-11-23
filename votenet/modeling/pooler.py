from typing import List

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from votenet.modeling.backbone import PointnetSAMoudleAgg
from votenet.structures import Instances, BoxMode


class ROIGridPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(self, grid_size, seed_feat_dim, rep_type, density):
        super().__init__()

        self.grid_size = grid_size
        self.density = density
        self.rep_type = rep_type
        if rep_type == "grid":
            self.num_key_points = grid_size ** 3
        elif rep_type == "ray":
            self.num_key_points = density * 6
        else:
            raise NotImplementedError

        self.seed_aggregation = PointnetSAMoudleAgg(
            radius=0.2,
            nsample=16,
            mlp=[seed_feat_dim, 128, 64, 32],
            use_xyz=True,
            normalize_xyz=True
        )

        # Reduce feature dim
        self.reduce_dim = torch.nn.Conv1d(self.num_key_points * 32, 128, 1)

        weight_init.c2_msra_fill(self.reduce_dim)

    def forward(self, seed_xyz: torch.Tensor, seed_features: torch.Tensor, proposals: List[Instances]):
        batch_size = len(proposals)
        # TODO: support different number of proposal_boxes
        num_proposals = len(proposals[0].proposal_boxes)
        dtype = seed_xyz.dtype
        device = seed_xyz.device

        # TODO: support different modes of boxes
        pred_box_reg = torch.stack(
            [x.proposal_boxes.get_tensor(assert_mode=BoxMode.XYZLBDRFU_ABS) for x in proposals]
        )
        pred_origins = torch.stack([x.proposal_boxes.get("origins") for x in proposals])
        if proposals[0].has("pred_heading_angles"):
            pred_heading_angles = torch.stack([x.pred_heading_angles for x in proposals])
        else:
            pred_heading_angles = torch.zeros(batch_size, num_proposals, dtype=dtype, device=device)

        if self.rep_type == "grid":
            # (bs, num_proposal, num_key_points, 3)
            key_points = get_global_grid_points_of_rois(
                pred_origins, pred_box_reg, pred_heading_angles, grid_size=self.grid_size
            )
        elif self.rep_type == "ray":
            key_points = convert_params_to_points(
                pred_origins, pred_box_reg, pred_heading_angles, density=self.density
            )
        else:
            raise NotImplementedError

        # (bs, num_proposal * num_key_points, 3)
        key_points = key_points.view(batch_size, -1, 3)

        features = self.seed_aggregation(seed_xyz, key_points, seed_features)
        # (bs, mlp[-1], num_proposal, num_key_points)
        features = features.view(batch_size, -1, num_proposals, self.num_key_points)
        # (bs, mlp[-1], num_key_points, num_proposal)
        features = features.transpose(2, 3)
        # (bs, mlp[-1]*num_key_points, num_proposal)
        features = features.reshape(batch_size, -1, num_proposals)
        # (bs, 128, num_proposal)
        features = self.reduce_dim(features)

        return features


def convert_params_to_points(center, rois, angle, density=1):
    batch_size, num_proposal, _ = rois.shape  # (B, N, 6)
    R = rotz_batch_pytorch(angle).reshape(-1, 3, 3)  # Rotation matrix ~ (B*N, 3, 3)
    # Convert param pairs (rois, angle) to point locations
    num_key_points = density * 6
    back_proj_points = torch.zeros((batch_size, num_proposal, 6, 3)).cuda()  # (B, N, 6, 3)
    back_proj_points[:, :, 0, 0] = - rois[:, :, 0]  # back
    back_proj_points[:, :, 1, 1] = - rois[:, :, 1]  # left
    back_proj_points[:, :, 2, 2] = - rois[:, :, 2]  # down
    back_proj_points[:, :, 3, 0] = rois[:, :, 3]  # front
    back_proj_points[:, :, 4, 1] = rois[:, :, 4]  # right
    back_proj_points[:, :, 5 ,2] = rois[:, :, 5]  # up
    back_proj_points = back_proj_points.reshape(-1, 6, 3)  # (B*N, 6, 3)
    local_points = [back_proj_points*float(i/density) for i in range(density, 0, -1)]
    local_points = torch.stack(local_points, dim=1)  # (B*N, density, 6, 3)
    local_points = local_points.transpose(1,2).contiguous()  # (B*N, 6, density, 3)
    local_points = local_points.reshape(-1, num_key_points, 3)  # (B*N, num_key_points, 3)
    local_points_rotated = torch.matmul(local_points, R)  # (B*N, num_key_points, 3)
    center = center.reshape(batch_size*num_proposal, 1, 3)  # (B*N, 1, 3)
    global_points = local_points_rotated + center  # (B*N, num_key_points, 3)
    global_points = global_points.reshape(batch_size, num_proposal, num_key_points, 3)  # (B, N, num_key_points, 3)

    return global_points


def get_global_grid_points_of_rois(center, rois, heading_angle, grid_size):
    """

    :param center:
    :param rois:
    :param heading_angle:
    :param grid_size:
    :return:
    """
    B = heading_angle.shape[0]
    N = heading_angle.shape[1]
    # Rotation matrix ~ (B*N, 3, 3)
    R = rotz_batch_pytorch(heading_angle.float()).view(-1, 3, 3)
    # (B*N, gs**3, 3)
    local_grid_points = get_dense_grid_points(rois, B*N, grid_size)
    # (B*N, gs**3, 3) ~ add Rotation
    local_grid_points = torch.matmul(local_grid_points, R)
    # (B*N, gs**3, 3)
    global_roi_grid_points = local_grid_points + center.view(B*N, 3).unsqueeze(1)
    # (B, N, gs**3, 3)
    global_roi_grid_points = global_roi_grid_points.view(B, N, -1, 3)
    return global_roi_grid_points


def rotz_batch_pytorch(t):
    """
    Rotation about z-axis
    :param t:
    :return:
    """
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3])).cuda()
    c = torch.cos(t)
    s = torch.sin(t)
    # Transposed rotation matrix for x'A' = (Ax)'
    # [[cos(t), -sin(t), 0],
    #  [sin(t), cos(t),  0],
    #  [0,      0,       1]]
    output[..., 0, 0] = c
    output[..., 0, 1] = -s
    output[..., 1, 0] = s
    output[..., 1, 1] = c
    output[..., 2, 2] = 1
    return output


def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
    """

    :param rois: (B, num_proposal, 6) ~ back/left/down/front/right/up
    :param batch_size_rcnn: B*num_proposal
    :param grid_size:
    :return:
    """
    faked_features = rois.new_ones((grid_size, grid_size, grid_size))  # alis gs for grid_size
    dense_idx = faked_features.nonzero(as_tuple=False)  # (gs**3, 3) [x_idx, y_idx, z_idx]
    dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (batch_size_rcnn, gs**3, 3)

    rois_center = rois[:, :, 0:3].view(-1, 3)  # (batch_size_rcnn, 3)
    local_rois_size = rois[:, :, 0:3] + rois[:, :, 3:6]  # (B, num_proposal, 3)
    local_rois_size = local_rois_size.view(-1, 3)  # (batch_size_rcnn, 3)
    roi_grid_points = (dense_idx + 0.5) / grid_size * local_rois_size.unsqueeze(dim=1)  # (batch_size_rcnn, gs**3, 3)
    roi_grid_points = roi_grid_points - rois_center.unsqueeze(dim=1)  # (batch_size_rcnn, gs**3, 3)
    return roi_grid_points

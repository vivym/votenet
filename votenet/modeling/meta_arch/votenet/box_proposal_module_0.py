import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from votenet.modeling.backbone import PointnetSAMoudleAgg


def build_box_proposal_module_0(cfg):
    num_heading_bin = cfg.MODEL.BOX_PROPOSAL_0.NUM_HEADING_BIN
    grid_size = cfg.MODEL.BOX_PROPOSAL_0.GRID_SIZE
    use_exp = cfg.MODEL.BOX_PROPOSAL_0.USE_EXP
    seed_feat_dim = cfg.MODEL.VOTING_MODULE.SEED_FEATURE_DIM

    return BoxProposalModule0(num_heading_bin, grid_size, seed_feat_dim, use_exp)


def decode_bbox_0(net, num_heading_bin, use_exp):
    net_transposed = net.transpose(2, 1).contiguous()  # (B, num_proposal, C)

    heading_scores = net_transposed[:, :, 0:num_heading_bin]
    heading_residuals_normalized = net_transposed[:, :, num_heading_bin:num_heading_bin*2]
    """
    end_points['heading_scores'] = heading_scores  # (B, num_proposal, num_heading_bin)
    end_points['heading_residuals_normalized'] = heading_residuals_normalized  # (B, num_proposal, num_heading_bin) in range[-1,1]
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi / num_heading_bin)  # (B, num_proposal, num_heading_bin)
    """

    rois = net_transposed[:, :, num_heading_bin*2: num_heading_bin*2+6]  # (B, num_proposal, 6)
    """
    if use_exp:
        end_points['rois_0'] = torch.exp(rois)
    else:
        end_points['rois_0'] = rois
    """

    centerness_scores = net_transposed[:, :, num_heading_bin*2+6:]  # (B, num_proposal, 1)
    # end_points['centerness_scores_0'] = centerness_scores.squeeze(2)
    return {
        "heading_scores": heading_scores,
        "heading_residuals_normalized": heading_residuals_normalized,
        "heading_residuals": heading_residuals_normalized * (np.pi / num_heading_bin),
        "rois_0": torch.exp(rois) if use_exp else rois,
        "centerness_scores_0": centerness_scores.squeeze(2),
    }


class BoxProposalModule0(nn.Module):
    def __init__(self, num_heading_bin, grid_size=3, seed_feat_dim=256, use_exp=True):
        super().__init__()

        self.num_heading_bin = num_heading_bin
        self.grid_size = grid_size
        self.use_exp = use_exp
        self.num_key_points = grid_size**3

        # object proposal 0
        self.output_channel = self.num_heading_bin*2 + 6 + 1 # 6 for back/left/down/front/right/up distance, 1 for center-ness score
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.output_channel, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        # ROI Grid pooling
        self.seed_aggregation = PointnetSAMoudleAgg(
            radius=0.2,
            nsample=16,
            mlp=[seed_feat_dim, 128, 64, 32],
            use_xyz=True,
            normalize_xyz=True
        )

        # Reduce feature dim
        self.reduce_dim = torch.nn.Conv1d(self.num_key_points*32, 128, 1)

    def forward(self, vote_xyz, vote_features, seed_xyz, seed_features):
        batch_size = vote_xyz.shape[0]
        num_proposal = vote_xyz.shape[1]

        # 1. Bbox_proposal_0 for orientation and 6 distances(back/left/down/front/right/up) and center-ness score
        net = F.relu(self.bn1(self.conv1(vote_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (B, num_heading_bin*2+6+1, num_proposal)
        results = decode_bbox_0(net, self.num_heading_bin, self.use_exp)

        # 2. Recover initial bbox
        # 2.1 recover heading angle
        pred_heading_class = torch.argmax(results["heading_scores"], -1)  # (B, num_proposal)
        pred_heading_residuals = torch.gather(
            results["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
        ).squeeze(-1)  # (B, num_proposal)

        if self.num_heading_bin != 1:  # for SUN RGBD dataset
            pred_heading = pred_heading_class.float() * (2 * np.pi / float(self.num_heading_bin)) + \
                           pred_heading_residuals  # (B, num_proposal)
            pred_heading = pred_heading % (2*np.pi)
        else:  # for ScanNetV2 dataset
            pred_heading = torch.zeros((batch_size, num_proposal)).cuda()
        results["heading_angle_0"] = pred_heading

        # 2.2 recover predicted distances
        pred_rois = results["rois_0"]  # (B, num_proposal, 6)

        # 3. Generate key points in rois_0 (for ROI Grid pooling)
        # (B, num_proposal, num_key_points, 3)
        key_points = get_global_grid_points_of_rois(vote_xyz, pred_rois, pred_heading, grid_size=self.grid_size)
        # (B, num_proposal*num_key_points, 3)
        key_points = key_points.view(batch_size, -1, 3)

        # 4. Aggregate seed points feature to key points
        # (B, mlp[-1], num_proposal*num_key_points)
        new_seed_features = self.seed_aggregation(seed_xyz, key_points, seed_features)
        # (B, mlp[-1], num_proposal, num_key_points)
        new_seed_features = new_seed_features.view(batch_size, -1, num_proposal, self.num_key_points)
        # (B, mlp[-1], num_key_points, num_proposal)
        new_seed_features = new_seed_features.transpose(2,3).contiguous()
        # (B, mlp[-1]*num_key_points, num_proposal)
        new_seed_features = new_seed_features.view(batch_size, -1, num_proposal)
        # (B, 128, num_proposal)
        new_seed_features = self.reduce_dim(new_seed_features)

        results["new_seed_features"] = new_seed_features

        return results


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
    output = torch.zeros(tuple(list(input_shape)+[3,3])).cuda()
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
    dense_idx = faked_features.nonzero()  # (gs**3, 3) [x_idx, y_idx, z_idx]
    dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (batch_size_rcnn, gs**3, 3)

    rois_center = rois[:, :, 0:3].view(-1, 3)  # (batch_size_rcnn, 3)
    local_rois_size = rois[:, :, 0:3] + rois[:, :, 3:6]  # (B, num_proposal, 3)
    local_rois_size = local_rois_size.view(-1, 3)  # (batch_size_rcnn, 3)
    roi_grid_points = (dense_idx+0.5) / grid_size * local_rois_size.unsqueeze(dim=1)  # (batch_size_rcnn, gs**3, 3)
    roi_grid_points = roi_grid_points - rois_center.unsqueeze(dim=1)  # (batch_size_rcnn, gs**3, 3)

    return roi_grid_points

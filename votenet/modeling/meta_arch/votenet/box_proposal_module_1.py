import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_box_proposal_module_1(cfg):
    num_heading_bin = cfg.MODEL.BOX_PROPOSAL_0.NUM_HEADING_BIN
    num_classes = cfg.MODEL.BOX_PROPOSAL_1.NUM_CLASSES
    use_centerness = cfg.MODEL.BOX_PROPOSAL_1.USE_CENTERNESS

    return BoxProposalModule1(num_heading_bin, num_classes, use_centerness=use_centerness)


def decode_bbox_1(net, proposal_0, num_class, use_centerness, num_heading_bin, reg_with_class):
    net_transposed = net.transpose(2, 1).contiguous()  # (batch_size, num_proposal, ...)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:, :, 0:2]
    # end_points['objectness_scores'] = objectness_scores  # (B, N, 2)

    heading_refined_normalized = net_transposed[:, :, 2:3].squeeze(2)  # (B, N)
    heading_refined_transform = "identical"
    if heading_refined_transform == "identical":
        heading_refined = proposal_0["heading_angle_0"] + heading_refined_normalized
    elif heading_refined_transform == "linear":
        heading_refined = proposal_0["heading_angle_0"] + heading_refined_normalized * (np.pi/num_heading_bin)
    else:
        raise NotImplementedError
    # end_points['heading_angle_1'] = heading_refined % (2*np.pi)

    sem_cls_scores = net_transposed[:, :, 3:3+num_class]  # (B, N, num_class)
    # end_points['sem_cls_scores'] = sem_cls_scores

    if reg_with_class:  # class sensitive
        rois_refined_output = net_transposed[:, :, 3+num_class:3+7*num_class]  # (B, N, 6*num_class)
    else:  # class agnostic
        rois_refined_output = net_transposed[:, :, 3+num_class:9+num_class]  # (B, N, 6)
    rois_refined_transform = "identical"
    if rois_refined_transform == "identical":
        rois_refined = rois_refined_output
    elif rois_refined_transform == "nonlinear":
        rois_refined = torch.tan(np.pi*(torch.sigmoid(rois_refined_output)-1/2))
    else:
        raise NotImplementedError
    if reg_with_class:  # recover rois_1 only for evaluation / use gt_sem_cls for training
        pred_sem_cls = torch.argmax(sem_cls_scores, -1).detach()  # (B, N)
        rois_refined = rois_refined.clone().detach()  # (B, N, 6*num_class)
        rois_refined = rois_refined.view(batch_size, num_proposal, 6, num_class)
        rois_refined = torch.gather(rois_refined, 3, pred_sem_cls.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 6, 1))
        rois_refined = rois_refined.squeeze(-1)  # (B, N, 6)
        rois_1 = proposal_0["rois_0"] + rois_refined
    else:
        rois_1 = proposal_0["rois_0"] + rois_refined

    if use_centerness:
        if reg_with_class:
            centerness_scores = net_transposed[:,:,3+7*num_class, 3+7*num_class+1]  # (B, N, 1)
        else:
            centerness_scores = net_transposed[:,:,9+num_class,9+num_class+1]  # (B, N, 1)
        # end_points['centerness_scores_1'] = centerness_scores.squeeze(2)  # (B, N)

    return {
        "objectness_scores": objectness_scores,
        "heading_angle_1": heading_refined % (2 * np.pi),
        "sem_cls_scores": sem_cls_scores,
        "rois_1": rois_1,
        "centerness_scores_1": centerness_scores.squeeze(2) if use_centerness else None
    }


class BoxProposalModule1(nn.Module):
    def __init__(self, num_heading_bin, num_class, fusion_type="concat", use_centerness=False):
        super().__init__()

        self.num_heading_bin = num_heading_bin
        self.num_class = num_class
        self.use_centerness = use_centerness

        # object proposal
        # Objectness scores (2)
        # heading residual (1)
        # size residual (6)
        # semantic classification (num_class)
        # centerness score (1, optional)
        if fusion_type == "concat":
            self.input_channel = 128 + 128
        elif fusion_type == "addition":
            self.input_channel = 128
        else:
            raise NotImplementedError

        self.output_channel = 2 + 1 + self.num_class

        self.reg_with_class = False
        if self.reg_with_class:  # class sensitive
            self.output_channel += 6*self.num_class
        else:  # class agnostic
            self.output_channel += 6

        if self.use_centerness:
            self.output_channel += 1

        self.conv1 = torch.nn.Conv1d(self.input_channel, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.output_channel, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, features, proposal_0):
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)

        results = decode_bbox_1(
            net, proposal_0, self.num_class, self.use_centerness, self.num_heading_bin, self.reg_with_class
        )

        return results

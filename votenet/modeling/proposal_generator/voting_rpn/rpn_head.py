import torch
from torch import nn

from votenet.config import configurable

from ..rpn_head import RPN_HEAD_REGISTRY


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, use_axis_aligned_box: bool, use_centerness: bool):
        super().__init__()

        self.use_axis_aligned_box = use_axis_aligned_box
        self.use_centerness = use_centerness

        self.convs = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
        )

        out_channels = 1 + 6  # objectness(1) + box_reg(6)
        if not use_axis_aligned_box:
            out_channels += 12 * 2
        if use_centerness:
            out_channels += 1

        self.predictor = nn.Conv1d(128, out_channels, kernel_size=1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "use_axis_aligned_box": cfg.INPUT.AXIS_ALIGNED_BOX,
            "use_centerness": cfg.MODEL.RPN.CENTERNESS,
        }

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): feature map

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        x = self.convs(x)
        x = self.predictor(x).permute(0, 2, 1)  # (bs, num_proposals, c)

        pred_objectness_logits = x[:, :, 0] # (bs, num_proposals)
        pred_box_reg = x[:, :, 1:7]
        idx = 7

        pred_heading_cls_logits, pred_heading_deltas = None, None
        if not self.use_axis_aligned_box:
            pred_heading_cls_logits = x[:, :, idx:idx + 12]
            pred_heading_deltas = x[:, :, idx + 12:idx + 24]

            idx += 24

        pred_centerness = None
        if self.use_centerness:
            pred_centerness = x[:, :, idx]

        return pred_objectness_logits, pred_box_reg, pred_heading_cls_logits, pred_heading_deltas, pred_centerness

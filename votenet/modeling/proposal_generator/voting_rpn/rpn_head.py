import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from votenet.config import configurable
from votenet.layers import build_positional_encoding_layer

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
    def __init__(
            self, *,
            use_axis_aligned_box: bool,
            use_centerness: bool,
            use_exp: bool,
            objectness_loss_type: str,
            fuse_xyz: int,
            fuse_xyz_type: str,
    ):
        super().__init__()

        self.use_axis_aligned_box = use_axis_aligned_box
        self.use_centerness = use_centerness
        self.use_exp = use_exp
        self.objectness_loss_type = objectness_loss_type
        self.fuse_xyz = fuse_xyz
        self.fuse_xyz_type = fuse_xyz_type

        if fuse_xyz > 0:
            self.positional_encoding = build_positional_encoding_layer(
                fuse_xyz_type, dim=fuse_xyz, fuse_type="cat"
            )
            num_extra_channels = self.positional_encoding.num_extra_channels
        else:
            num_extra_channels = 0

        convs = [
            nn.Conv1d(128 + num_extra_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        ]
        self.convs = nn.Sequential(*convs)

        self.objectness_predictor = nn.Conv1d(
            128, 2 if objectness_loss_type == "cross_entropy" else 1,
            kernel_size=1,
        )
        self.box_predictor = nn.Conv1d(128, 6, kernel_size=1)
        predictors = [self.objectness_predictor, self.box_predictor]
        # TODO:
        assert use_axis_aligned_box
        if use_centerness:
            self.centerness_predictor = nn.Conv1d(128, 1, kernel_size=1)
            predictors.append(self.centerness_predictor)

        for layer in convs:
            if isinstance(layer, nn.Conv1d):
                weight_init.c2_msra_fill(layer)

        for predictor in predictors:
            nn.init.normal_(predictor.weight, std=0.001)
            if predictor.bias is not None:
                nn.init.constant_(predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        return {
            "use_axis_aligned_box": cfg.INPUT.AXIS_ALIGNED_BOX,
            "use_centerness": cfg.MODEL.RPN.CENTERNESS,
            "use_exp": cfg.MODEL.RPN.USE_EXP,
            "objectness_loss_type": cfg.MODEL.RPN.OBJECTNESS_LOSS_TYPE,
            "fuse_xyz": cfg.MODEL.RPN.FUSE_XYZ,
            "fuse_xyz_type": cfg.MODEL.RPN.FUSE_XYZ_TYPE,
        }

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        Args:
            xyz(Tensor): xyz
            features (Tensor): feature map

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        if self.fuse_xyz > 0:
            features = self.positional_encoding(xyz.permute(0, 2, 1), features)

        x = self.convs(features)

        # (bs, num_proposals, 2)
        pred_objectness_logits = self.objectness_predictor(x).permute(0, 2, 1)
        # (bs, num_proposals, 6)
        pred_box_reg = self.box_predictor(x).permute(0, 2, 1)
        if self.use_exp:
            pred_box_reg = pred_box_reg.exp()

        pred_heading_cls_logits, pred_heading_deltas = None, None

        # TODO:
        assert self.use_axis_aligned_box

        pred_centerness = None
        if self.use_centerness:
            # (bs, num_proposals)
            pred_centerness = self.centerness_predictor(x).squeeze(1)

        return pred_objectness_logits, pred_box_reg, pred_heading_cls_logits, pred_heading_deltas, pred_centerness

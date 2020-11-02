from torch import nn

from votenet.config import configurable
from votenet.utils.registry import Registry

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_box_head(cfg):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg)


@ROI_BOX_HEAD_REGISTRY.register()
class StandardBoxHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(self, *, num_classes: int, use_axis_aligned_box: bool, use_centerness: bool):
        super().__init__()

        self.num_classes = num_classes
        self.use_axis_aligned_box = use_axis_aligned_box
        self.use_centerness = use_centerness

        self.convs = nn.Sequential(
            nn.Conv1d(128 + 128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
        )

        out_channels = (num_classes + 1) + (num_classes + 1) * 6
        if not use_axis_aligned_box:
            out_channels += (num_classes + 1) * 1
        if use_centerness:
            out_channels += 1

        self.cls_predictor = nn.Conv1d(128, num_classes + 1, kernel_size=1)
        self.box_predictor = nn.Conv1d(128, (num_classes + 1) * 6, kernel_size=1)
        if not use_axis_aligned_box:
            self.box_angle_predictor = nn.Conv1d(128, (num_classes + 1) * 6, kernel_size=1)
        if use_centerness:
            self.centerness_predictor = nn.Conv1d(128, 1, kernel_size=1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "use_axis_aligned_box": cfg.INPUT.AXIS_ALIGNED_BOX,
            "use_centerness": cfg.MODEL.ROI_HEADS.CENTERNESS,
        }

    def forward(self, x):
        batch_size = x.size(0)
        num_proposals = x.size(-1)

        x = self.convs(x)
        # x = self.predictor(x).permute(0, 2, 1)  # (bs, num_proposals, c)

        pred_cls_logits = self.cls_predictor(x)
        pred_box_deltas = self.box_predictor(x).permute(0, 2, 1).view(
            batch_size, num_proposals, self.num_classes + 1, 6
        )

        pred_heading_deltas = None
        if not self.use_axis_aligned_box:
            pred_heading_deltas = self.box_angle_predictor(x).permute(0, 2, 1)

        pred_centerness = None
        if self.use_centerness:
            pred_centerness = self.centerness_predictor(x).permute(0, 2, 1)

        return pred_cls_logits, pred_box_deltas, pred_heading_deltas, pred_centerness

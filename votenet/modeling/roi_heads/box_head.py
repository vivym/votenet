import fvcore.nn.weight_init as weight_init
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
    def __init__(
            self, *, num_classes: int, use_axis_aligned_box: bool, use_centerness: bool, use_exp: bool
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_axis_aligned_box = use_axis_aligned_box
        self.use_centerness = use_centerness
        self.use_exp = use_exp

        convs = [
            nn.Conv1d(128 + 128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        ]
        self.convs = nn.Sequential(*convs)

        self.cls_predictor = nn.Conv1d(128, num_classes + 1, kernel_size=1)
        self.box_predictor = nn.Conv1d(128, num_classes * 6, kernel_size=1)
        predictors = [self.cls_predictor, self.box_predictor]
        if not use_axis_aligned_box:
            self.box_angle_predictor = nn.Conv1d(128, num_classes * 1, kernel_size=1)
            predictors.append(self.box_angle_predictor)
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
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "use_axis_aligned_box": cfg.INPUT.AXIS_ALIGNED_BOX,
            "use_centerness": cfg.MODEL.ROI_HEADS.CENTERNESS,
            "use_exp": cfg.MODEL.ROI_BOX_HEAD.USE_EXP,
        }

    def forward(self, x):
        batch_size = x.size(0)
        num_proposals = x.size(-1)

        x = self.convs(x)

        pred_cls_logits = self.cls_predictor(x)
        pred_box_deltas = self.box_predictor(x).permute(0, 2, 1).view(
            batch_size, num_proposals, self.num_classes, 6
        )  # (bs, num_proposals, num_classes, 6)
        if self.use_exp:
            pred_box_deltas = pred_box_deltas.exp()

        pred_heading_deltas = None
        if not self.use_axis_aligned_box:
            # (bs, num_proposals, num_classes)
            pred_heading_deltas = self.box_angle_predictor(x).permute(0, 2, 1)

        pred_centerness = None
        if self.use_centerness:
            # (bs, num_proposals)
            pred_centerness = self.centerness_predictor(x).view(batch_size, num_proposals)

        return pred_cls_logits, pred_box_deltas, pred_heading_deltas, pred_centerness

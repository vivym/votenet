from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from votenet.utils.events import get_event_storage
from votenet.structures import Instances, Boxes, BoxMode


class BoxOutputLayer(nn.Module):
    """
    An internal implementation that stores information about outputs of a box head,
    and provides methods that are used to decode the outputs of a box head.
    """

    def __init__(
        self,
        pred_class_logits: torch.Tensor,
        pred_proposal_deltas: torch.Tensor,
        proposals: List[Instances],
    ):
        super().__init__()

        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.get_tensor().requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes.from_tensor(
                torch.zeros(0, 7, device=self.pred_proposal_deltas.device), mode=BoxMode.XYZWDH_ABS
            )
        self._no_instances = len(proposals) == 0  # no instances found

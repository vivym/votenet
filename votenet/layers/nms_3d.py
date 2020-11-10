from typing import List

import torch

from votenet import _C


def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Parameters
    ----------
    boxes : Tensor[N, 6])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, z1, x2, y2, z2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return _C.nms_3d(boxes, scores, iou_threshold)


def batched_nms_3d(
        boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 6]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, z1, x2, y2, z2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    assert boxes.shape[-1] == 6

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    else:
        # TODO may need better strategy.
        # Investigate after having a fully-cuda NMS op.
        if len(boxes) < 40000:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]
            keep = nms_3d(boxes_for_nms, scores, iou_threshold)
            return keep
        else:
            result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
                mask = (idxs == id).nonzero().view(-1)
                keep = nms_3d(boxes[mask], scores[mask], iou_threshold)
                result_mask[mask[keep]] = True
            keep = result_mask.nonzero().view(-1)
            keep = keep[scores[keep].argsort(descending=True)]
            return keep

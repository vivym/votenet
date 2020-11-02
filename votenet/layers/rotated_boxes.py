import torch

from votenet import _C


def pairwise_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    Arguments:
        boxes1 (Tensor[N, 5])
        boxes2 (Tensor[M, 5])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    return _C.box_iou_rotated(boxes1, boxes2)


def pairwise_3d_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, z_center, width, depth, height, angle) format.

    Arguments:
        boxes1 (Tensor[N, 7])
        boxes2 (Tensor[M, 7])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    z1, h1, a1 = boxes1[:, 2], boxes1[:, 5] / 2., boxes1[:, 6]
    z2, h2, a2 = boxes2[:, 2], boxes2[:, 5] / 2., boxes2[:, 6]

    boxes_2d_1 = torch.cat([
        boxes1[:, :2],
        boxes1[:, 3:5],
        a1.view(-1, 1),
    ])

    boxes_2d_2 = torch.cat([
        boxes2[:, :2],
        boxes2[:, 3:5],
        a2.view(-1, 1),
    ])

    iou = pairwise_iou_rotated(boxes_2d_1, boxes_2d_2)

    p1 = torch.max(z1 - h1, z2 - h2, dim=1)
    p2 = torch.min(z1 + h1, z2 + h2, dim=1)
    iou_z = (p2 - p1).clamp(min=0.)

    return iou * iou_z

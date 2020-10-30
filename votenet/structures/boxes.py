from enum import IntEnum, unique
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import device

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYZXYZ_ABS = 0
    """
    (x0, y0, z0, x1, y1, z1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYZWDH_ABS = 1
    """
    (x0, y0, z0, w, d, h) in absolute floating points coordinates.
    """
    XYZXYZ_REL = 2
    """
    Not yet supported!
    (x0, y0, z0, x1, y1, z1) in range [0, 1]. They are relative to the size of the image.
    """
    XYZWDH_REL = 3
    """
    Not yet supported!
    (x0, y0, z0, w, d, h) in range [0, 1]. They are relative to the size of the image.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 6
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 6, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 6"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode.value not in [
            BoxMode.XYZXYZ_REL,
            BoxMode.XYZWDH_REL,
        ] and from_mode.value not in [
            BoxMode.XYZXYZ_REL,
            BoxMode.XYZWDH_REL,
        ], "Relative mode not yet supported!"

        if to_mode == BoxMode.XYZXYZ_ABS and from_mode == BoxMode.XYZWDH_ABS:
            half_width = arr[:, 3] / 2.
            half_depth = arr[:, 4] / 2.
            half_height = arr[:, 5] / 2.
            arr[:, 3] = arr[:, 0] + half_width
            arr[:, 4] = arr[:, 1] + half_depth
            arr[:, 5] = arr[:, 2] + half_height
            arr[:, 0] -= half_width
            arr[:, 1] -= half_depth
            arr[:, 2] -= half_height
        elif from_mode == BoxMode.XYZXYZ_ABS and to_mode == BoxMode.XYZWDH_ABS:
            width = arr[:, 3] - arr[:, 0]
            depth = arr[:, 4] - arr[:, 1]
            height = arr[:, 5] - arr[:, 2]
            arr[:, 0] += width / 2.
            arr[:, 1] += depth / 2.
            arr[:, 2] += height / 2.
            arr[:, 3] = width
            arr[:, 4] = depth
            arr[:, 5] = height
        else:
            raise NotImplementedError(
                "Conversion from BoxMode {} to {} is not supported yet".format(
                    from_mode, to_mode
                )
            )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class Boxes:
    """
    This structure stores a list of boxes as a Nx6 torch.Tensor.
    It supports some common methods about boxes
    (`volume`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx6. Each row is (x1, y1, z1, x2, y2, z2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx6 matrix.  Each row is (x1, y1, z1, x2, y2, z2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 6)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 6, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any):
        return Boxes(self.tensor.to(*args, **kwargs))

    def volume(self) -> torch.Tensor:
        """
        Computes the volumes of all the boxes.

        Returns:
            torch.Tensor: a vector with volumes of each box.
        """
        box = self.tensor
        volume = (box[:, 3] - box[:, 0]) * (box[:, 4] - box[:, 1]) * (box[:, 5] - box[:, 2])
        return volume

    def clip(self, box_size: Tuple[int, int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width],
        y coordinates to the range [0, depth] and z coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        w, d, h = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=d)
        self.tensor[:, 2].clamp_(min=0, max=h)
        self.tensor[:, 3].clamp_(min=0, max=w)
        self.tensor[:, 4].clamp_(min=0, max=d)
        self.tensor[:, 5].clamp_(min=0, max=h)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 3] - box[:, 0]
        depths = box[:, 4] - box[:, 1]
        heights = box[:, 5] - box[:, 2]
        keep = (widths > threshold) & (depths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item):
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (width, depth, height): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        width, depth, height = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] >= -boundary_threshold)
            & (self.tensor[..., 3] < width + boundary_threshold)
            & (self.tensor[..., 4] < depth + boundary_threshold)
            & (self.tensor[..., 5] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx3 array of (x, y, z).
        """
        return (self.tensor[:, :3] + self.tensor[:, 3:]) / 2

    def scale(self, scale_x: float, scale_y: float, scale_z: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::3] *= scale_x
        self.tensor[:, 1::3] *= scale_y
        self.tensor[:, 2::3] *= scale_z

    # classmethod not supported by torchscript. TODO try staticmethod
    @classmethod
    @torch.jit.unused
    def cat(cls, boxes_list):
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Similar to pariwise_iou but compute the IoA (intersection over boxes2 area).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = torch.where(
        inter > 0, inter / area2, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix

    Args:
        boxes1: (Boxes) bounding boxes, sized [N,4].
        boxes2: (Boxes) bounding boxes, sized [N,4].
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou

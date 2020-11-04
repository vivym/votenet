from enum import IntEnum, unique
from typing import Any, List, Tuple, Union

import numpy as np
import torch

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
    (xc, yc, zc, w, d, h) in absolute floating points coordinates.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 6 or 7
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
            assert len(box) == 6 or len(box) == 7, (
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

        if to_mode == BoxMode.XYZXYZ_ABS and from_mode == BoxMode.XYZWDH_ABS:
            half_sizes = arr[:, 3:6] / 2.
            arr[:, 3:6] = arr[:, :3] + half_sizes
            arr[:, :3] -= half_sizes
        elif from_mode == BoxMode.XYZXYZ_ABS and to_mode == BoxMode.XYZWDH_ABS:
            sizes = arr[:, 3:6] - arr[:, :3]
            arr[:, :3] += sizes / 2.
            arr[:, 3:6] = sizes
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
    This structure stores a list of boxes as a Nx6/7 torch.Tensor.
    It supports some common methods about boxes
    (`volume`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx6/7.
        Each row is (x0, y0, z0, x1, y1, z1, angle).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nxk matrix, where k = 6 or 7.
            Each row is (x0, y0, z0, x1, y1, z1, angle).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 7)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and (tensor.size(-1) == 7 or tensor.size(-1) == 6), tensor.size()

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
        # TODO: support angle
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
        return (self.tensor[:, :3] + self.tensor[:, 3:6]) / 2

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
    def device(self) -> torch.device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (6/7,) at a time.
        """
        yield from self.tensor

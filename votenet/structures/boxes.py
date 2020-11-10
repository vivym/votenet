from abc import ABCMeta, abstractmethod
from enum import IntEnum, unique
from typing import List, Any, Tuple, Union, Optional

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
    """
    XYZWDH_ABS = 1
    """
    (xc, yc, zc, w, d, h) in absolute floating points coordinates.
    """
    XYZLBDRFU_ABS = 2
    """
    (x0, y0, z0, left, back, down, right, front, up) in absolute floating points coordinates.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode", **kwargs: Any) -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 6 or 7 or 9
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
            box = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                box = torch.from_numpy(np.asarray(box))

        box_dim = box.size(-1)
        if from_mode == BoxMode.XYZWDH_ABS or from_mode == BoxMode.XYZXYZ_ABS:
            assert box_dim == 6 or box_dim == 7, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 6 or 7"
            )
        else:
            assert box_dim == 9 or box_dim == 10, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 9 or 10"
            )

        if from_mode == BoxMode.XYZWDH_ABS:
            has_angles = box.size(-1) == 7
            if to_mode == BoxMode.XYZXYZ_ABS:
                half_sizes = box[:, 3:6] / 2.
                centers = box[:, 0:3]

                arr = torch.zeros_like(box, dtype=box.dtype, device=box.device)
                arr[:, 0:3] = centers - half_sizes
                arr[:, 3:6] = centers + half_sizes
                if has_angles:
                    arr[:, 6] = box[:, 6]
            else:  # BoxMode.XYZLBDRFU_ABS
                # TODO: consider rotations
                assert "origins" in kwargs
                origins = kwargs["origins"]
                half_sizes = box[:, 3:6] / 2.
                p1 = box[:, 0:3] - half_sizes
                p2 = box[:, 0:3] + half_sizes

                arr = torch.zeros(box.size(0), 10 if has_angles else 9, dtype=box.dtype, device=box.device)
                arr[:, 0:3] = origins
                arr[:, 3:6] = origins - p1
                arr[:, 6:9] = p2 - origins
                if has_angles:
                    arr[:, 9] = box[:, 6]
        elif from_mode == BoxMode.XYZXYZ_ABS:
            has_angles = box.size(-1) == 7
            if to_mode == BoxMode.XYZWDH_ABS:
                sizes = box[:, 3:6] - box[:, 0:3]

                arr = torch.zeros_like(box, dtype=box.dtype, device=box.device)
                arr[:, 0:3] = box[:, 0:3] + sizes / 2.
                arr[:, 3:6] = sizes
                if has_angles:
                    arr[:, 6] = box[:, 6]
            else:  # BoxMode.XYZLBDRFU_ABS
                # TODO: consider rotations
                assert "origins" in kwargs
                origins = kwargs["origins"]
                p1 = box[:, 0:3]
                p2 = box[:, 3:6]

                arr = torch.zeros(box.size(0), 10 if has_angles else 9, dtype=box.dtype, device=box.device)
                arr[:, 0:3] = origins
                arr[:, 3:6] = origins - p1
                arr[:, 6:9] = p2 - origins
                if has_angles:
                    arr[:, 9] = box[:, 6]
        else:  # BoxMode.XYZLBDRFU_ABS
            has_angles = box.size(-1) == 10
            # TODO: consider rotations
            origins = box[:, 0:3]
            lbd = box[:, 3:6]
            rfu = box[:, 6:9]
            p1 = origins - lbd
            p2 = origins + rfu

            arr = torch.zeros(box.size(0), 7 if has_angles else 6, dtype=box.dtype, device=box.device)
            if to_mode == BoxMode.XYZWDH_ABS:
                sizes = p2 - p1
                arr[:, 0:3] = p1 + sizes / 2.
                arr[:, 3:6] = sizes
            else:  # BoxMode.XYZXYZ_ABS
                arr[:, 0:3] = p1
                arr[:, 3:6] = p2
            if has_angles:
                arr[:, 6] = box[:, 9]

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class Boxes(object, metaclass=ABCMeta):
    def __init__(self, tensor: torch.Tensor, mode: "BoxMode" = BoxMode.XYZWDH_ABS):
        self._tensor = tensor
        self._mode = mode

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, mode: "BoxMode") -> "Boxes":
        if mode == BoxMode.XYZWDH_ABS:
            return XYZWDHBoxes(tensor)
        elif mode == BoxMode.XYZXYZ_ABS:
            return XYZXYZBoxes(tensor)
        elif mode == BoxMode.XYZLBDRFU_ABS:
            return XYZLBDRFUBoxes(tensor)
        else:
            raise NotImplementedError

    def convert(self, to_mode: "BoxMode", **kwargs: Any) -> "Boxes":
        if to_mode == self._mode:
            return self
        tensor = self.get_tensor(to_mode, **kwargs)
        return Boxes.from_tensor(tensor, mode=to_mode)

    def get_tensor(self, mode: Optional["BoxMode"] = None, **kwargs: Any) -> torch.Tensor:
        if mode is None or mode == self._mode:
            return self._tensor
        else:
            return BoxMode.convert(
                self._tensor, from_mode=self._mode, to_mode=mode, **kwargs
            )

    @property
    def mode(self) -> "BoxMode":
        return self._mode

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return type(self)(self._tensor.clone())

    def to(self, *args: Any, **kwargs: Any) -> "Boxes":
        return type(self)(self._tensor.to(*args, **kwargs))

    @property
    @abstractmethod
    def with_angle(self) -> bool:
        pass

    @abstractmethod
    def get_centers(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_sizes(self) -> torch.Tensor:
        # TODO: consider rotation
        pass

    def volume(self) -> torch.Tensor:
        return self.get_sizes().prod(dim=-1)

    def __getitem__(self, item) -> "Boxes":
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
        cls = type(self)
        if isinstance(item, int):
            return cls(self._tensor[item].view(1, -1))
        b = self._tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return cls(b)

    def __len__(self) -> int:
        return self._tensor.shape[0]

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @classmethod
    @torch.jit.unused
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all([isinstance(box, Boxes) for box in boxes_list])
        assert all([type(box) == type(boxes_list[0]) for box in boxes_list])
        type_cls = type(boxes_list[0])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = type_cls(torch.cat([b._tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    def __iter__(self):
        """
        Yield a box as a Tensor of shape (6/7,) at a time.
        """
        yield from self._tensor


class XYZXYZBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nxk matrix, where k = 6 or 7.
            Each row is (x0, y0, z0, x1, y1, z1, [angle]).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 7)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and (tensor.size(-1) == 7 or tensor.size(-1) == 6), tensor.size()

        super().__init__(tensor, mode=BoxMode.XYZXYZ_ABS)

    @property
    def with_angle(self) -> bool:
        return self._tensor.size(-1) == 7

    def get_centers(self) -> torch.Tensor:
        return (self._tensor[:, 0:3] + self._tensor[:, 3:6]) / 2.

    def get_sizes(self) -> torch.Tensor:
        return self._tensor[:, 3:6] - self._tensor[:, 0:3]

    def __repr__(self) -> str:
        return f"XYZXYZBoxes({str(self._tensor)})"


class XYZWDHBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nxk matrix, where k = 6 or 7.
            Each row is (xc, yc, zc, width, depth, height, [angle]).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 7)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and (tensor.size(-1) == 7 or tensor.size(-1) == 6), tensor.size()

        super().__init__(tensor, mode=BoxMode.XYZWDH_ABS)

    @property
    def with_angle(self) -> bool:
        return self._tensor.size(-1) == 7

    def get_centers(self) -> torch.Tensor:
        return self._tensor[:, 0:3]

    def get_sizes(self) -> torch.Tensor:
        return self._tensor[:, 3:6]

    def __repr__(self) -> str:
        return f"XYZWDHBoxes({str(self._tensor)})"


class XYZLBDRFUBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor, angles: Optional[torch.Tensor] = None):
        """
        Args:
            tensor (Tensor[float]): a Nxk matrix, where k = 9 or 10.
            Each row is (xc, yc, zc, width, depth, height, [angle]).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 9)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and (tensor.size(-1) == 10 or tensor.size(-1) == 9), tensor.size()

        super().__init__(tensor, mode=BoxMode.XYZLBDRFU_ABS)

    @property
    def with_angle(self) -> bool:
        return self._tensor.size(-1) == 10

    def get_centers(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_centers()

    def get_sizes(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_sizes()

    def __repr__(self) -> str:
        return f"XYZLBDRFUBoxes({str(self._tensor)})"


def pairwise_intersection(boxes1: "Boxes", boxes2: "Boxes") -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N, M].
    """
    boxes1 = boxes1.get_tensor(BoxMode.XYZXYZ_ABS)
    boxes2 = boxes2.get_tensor(BoxMode.XYZXYZ_ABS)
    width_depth_height = torch.min(boxes1[:, None, 3:6], boxes2[:, 3:6]) - torch.max(
        boxes1[:, None, 0:3], boxes2[:, 0:3]
    )  # [N, M, 3]
    width_depth_height.clamp_(min=0)  # [N, M, 3]
    intersection = width_depth_height.prod(dim=-1)  # [N, M]
    return intersection


def pairwise_iou(boxes1: "Boxes", boxes2: "Boxes") -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N, M].
    """
    vol1 = boxes1.volume()  # [N]
    vol2 = boxes2.volume()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (vol1[:, None] + vol2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

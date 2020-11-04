from abc import ABCMeta, abstractmethod
from enum import IntEnum, unique
from typing import List, Any, Dict, Tuple, Union, Optional

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
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode", **kwargs: Dict[str, Any]) -> _RawBoxType:
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
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

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

        if from_mode == BoxMode.XYZWDH_ABS and to_mode == BoxMode.XYZXYZ_ABS:
            half_sizes = arr[:, 3:6] / 2.
            arr[:, 3:6] = arr[:, 0:3] + half_sizes
            arr[:, 0:3] -= half_sizes
        elif from_mode == BoxMode.XYZXYZ_ABS and to_mode == BoxMode.XYZWDH_ABS:
            sizes = arr[:, 3:6] - arr[:, 0:3]
            arr[:, 0:3] += sizes / 2.
            arr[:, 3:6] = sizes
        elif from_mode == BoxMode.XYZWDH_ABS and to_mode == BoxMode.XYZLBDRFU_ABS:
            assert "origins" in kwargs
            origins = kwargs["origins"]
            has_angles = arr.size(0) == 7
            half_sizes = arr[:, 3:6] / 2.
            p1 = arr[:, 0:3] - half_sizes
            p2 = arr[:, 0:3] + half_sizes
            angles = arr[:, 6] if has_angles else None
            arr = torch.zeros(arr.size(0), 10 if has_angles else 9, dtype=arr.dtype, device=arr.device)
            if has_angles:
                arr[:, 9] = angles
            arr[:, 0:3] = origins
            arr[:, 3:6] = origins - p1
            arr[:, 6:9] = p2 - origins
        elif from_mode == BoxMode.XYZLBDRFU_ABS and \
                (to_mode == BoxMode.XYZXYZ_ABS or to_mode == BoxMode.XYZWDH_ABS):
            has_angles = arr.size(0) == 10
            xyz = arr[:, 0:3]
            lbd = arr[:, 3:6]
            rfu = arr[:, 6:9]
            angles = arr[:, 9] if has_angles else None
            arr = torch.zeros(arr.size(0), 7 if has_angles else 6, dtype=arr.dtype, device=arr.device)
            if has_angles:
                arr[:, 6] = angles
            p1 = xyz - lbd
            p2 = xyz + rfu

            if to_mode == BoxMode.XYZWDH_ABS:
                sizes = p2 - p1
                arr[:, 0:3] += sizes / 2.
                arr[:, 3:6] = sizes
            else:
                arr[:, 0:3] = p1
                arr[:, 3:6] = p2
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


class Boxes(object, metaclass=ABCMeta):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, mode: "BoxMode") -> "Boxes":
        if mode == BoxMode.XYZWDH_ABS:
            return XYZWDHBoxes(tensor)
        elif mode == BoxMode.XYZLBDRFU_ABS:
            return XYZLBDRFUBoxes(tensor)
        else:
            raise NotImplementedError

    @abstractmethod
    def convert(self, to_mode: "BoxMode"):
        pass

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return type(self)(self.tensor.clone())

    def to(self, *args: Any, **kwargs: Any):
        return type(self)(self.tensor.to(*args, **kwargs))

    @property
    @abstractmethod
    def with_angle(self) -> bool:
        pass

    @abstractmethod
    def get_centers(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_sizes(self) -> torch.Tensor:
        pass

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
        cls = type(self)
        if isinstance(item, int):
            return cls(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return cls(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

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
        cat_boxes = type_cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self):
        """
        Yield a box as a Tensor of shape (6/7,) at a time.
        """
        yield from self.tensor


class XYZWDHBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor, angles: Optional[torch.Tensor] = None):
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

        super().__init__(tensor)

    def convert(self, to_mode: "BoxMode", **kwargs: Dict[str, Any]):
        if to_mode == BoxMode.XYZLBDRFU_ABS:
            assert "origins" in kwargs
            boxes = BoxMode.convert(
                self.tensor, from_mode=BoxMode.XYZWDH_ABS, to_mode=BoxMode.XYZLBDRFU_ABS, **kwargs
            )
            return XYZLBDRFUBoxes(boxes)
        else:
            raise NotImplementedError

    @property
    def with_angle(self) -> bool:
        return self.tensor.size(-1) == 7

    def get_centers(self) -> torch.Tensor:
        return self.tensor[:, 0:3]

    def get_sizes(self) -> torch.Tensor:
        return self.tensor[:, 3:6]

    def __repr__(self) -> str:
        return f"XYZWDHBoxes({str(self.tensor)})"


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

        super().__init__(tensor)

    def convert(self, to_mode: "BoxMode"):
        if to_mode == BoxMode.XYZWDH_ABS:
            boxes = BoxMode.convert(
                self.tensor, from_mode=BoxMode.XYZLBDRFU_ABS, to_mode=BoxMode.XYZWDH_ABS
            )
            return XYZWDHBoxes(boxes)
        else:
            raise NotImplementedError

    @property
    def with_angle(self) -> bool:
        return self.tensor.size(-1) == 10

    def get_centers(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_centers()

    def get_sizes(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_sizes()

    def __repr__(self) -> str:
        return f"XYZLBDRFUBoxes({str(self.tensor)})"

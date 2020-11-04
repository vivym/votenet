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
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
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
        elif from_mode == BoxMode.XYZLBDRFU_ABS and \
                (to_mode == BoxMode.XYZXYZ_ABS or to_mode == BoxMode.XYZWDH_ABS):
            xyz = arr[:, 0:3]
            lbd = arr[:, 3:6]
            rfu = arr[:, 6:9]
            arr = torch.zeros(arr.size(0), 6, dtype=arr.dtype, device=arr.device)
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

    @abstractmethod
    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        pass

    @abstractmethod
    def to(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def get_centers(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_sizes(self) -> torch.Tensor:
        pass


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
            tensor = tensor.reshape((0, 6)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and (tensor.size(-1) == 7 or tensor.size(-1) == 6), tensor.size()

        self.boxes = tensor[:, :6]
        if angles is not None:
            assert self.boxes.size(0) == self.angles.size(0)
            self.angles = angles
        else:
            self.angles = tensor[:, 6] if tensor.size(-1) == 7 else None

    def convert(self, to_mode: "BoxMode"):
        if to_mode == BoxMode.XYZLBDRFU_ABS:
            boxes = BoxMode.convert(
                self.boxes, from_mode=BoxMode.XYZWDH_ABS, to_mode=BoxMode.XYZLBDRFU_ABS
            )
            return XYZLBDRFUBoxes(boxes, self.angles)
        else:
            raise NotImplementedError

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return XYZWDHBoxes(self.boxes.clone(), None if self.angles is None else self.angles.clone())

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any):
        return XYZWDHBoxes(
            self.boxes.to(*args, **kwargs),
            None if self.angles is None else self.angles.to(*args, **kwargs),
        )

    def get_centers(self) -> torch.Tensor:
        return self.boxes[: 0:3]

    def get_sizes(self) -> torch.Tensor:
        return self.boxes[:, 3:6]


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

        self.boxes = tensor[:, :9]
        if angles is not None:
            assert self.boxes.size(0) == self.angles.size(0)
            self.angles = angles
        else:
            self.angles = tensor[:, 9] if tensor.size(-1) == 10 else None

    def convert(self, to_mode: "BoxMode"):
        if to_mode == BoxMode.XYZWDH_ABS:
            boxes = BoxMode.convert(
                self.boxes, from_mode=BoxMode.XYZLBDRFU_ABS, to_mode=BoxMode.XYZWDH_ABS
            )
            return XYZWDHBoxes(boxes, self.angles)
        else:
            raise NotImplementedError

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return XYZLBDRFUBoxes(self.boxes.clone(), None if self.angles is None else self.angles.clone())

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any):
        return XYZLBDRFUBoxes(
            self.boxes.to(*args, **kwargs),
            None if self.angles is None else self.angles.to(*args, **kwargs),
        )

    def get_centers(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_centers()

    def get_sizes(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_sizes()

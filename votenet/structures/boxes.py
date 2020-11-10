from abc import ABCMeta, abstractmethod
from enum import IntEnum, unique
from typing import Dict, List, Any, Tuple, Union, Optional
import itertools

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
    (left, back, down, right, front, up) in absolute floating points coordinates.
    """

    @staticmethod
    def convert(
            box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode", **kwargs: Any
    ) -> Tuple[_RawBoxType, Dict]:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 6
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box, kwargs

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
        assert box_dim == 6, (
            "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
            " where k == 6"
        )

        arr = torch.zeros_like(box, dtype=box.dtype, device=box.device)
        if from_mode == BoxMode.XYZWDH_ABS:
            has_angles = "angles" in kwargs
            if to_mode == BoxMode.XYZXYZ_ABS:
                half_sizes = box[:, 3:6] / 2.
                centers = box[:, 0:3]

                arr[:, 0:3] = centers - half_sizes
                arr[:, 3:6] = centers + half_sizes
            else:  # BoxMode.XYZLBDRFU_ABS
                # TODO: consider rotations
                assert "origins" in kwargs
                origins = kwargs["origins"]
                half_sizes = box[:, 3:6] / 2.
                p1 = box[:, 0:3] - half_sizes
                p2 = box[:, 0:3] + half_sizes

                arr[:, 0:3] = origins - p1
                arr[:, 3:6] = p2 - origins
        elif from_mode == BoxMode.XYZXYZ_ABS:
            has_angles = "angles" in kwargs
            if to_mode == BoxMode.XYZWDH_ABS:
                sizes = box[:, 3:6] - box[:, 0:3]

                arr[:, 0:3] = box[:, 0:3] + sizes / 2.
                arr[:, 3:6] = sizes
            else:  # BoxMode.XYZLBDRFU_ABS
                # TODO: consider rotations
                assert "origins" in kwargs
                origins = kwargs["origins"]
                p1 = box[:, 0:3]
                p2 = box[:, 3:6]

                arr[:, 0:3] = origins - p1
                arr[:, 3:6] = p2 - origins
        else:  # BoxMode.XYZLBDRFU_ABS
            has_angles = "angles" in kwargs
            # TODO: consider rotations
            assert "origins" in kwargs
            origins = kwargs["origins"]
            lbd = box[:, 0:3]
            rfu = box[:, 3:6]
            p1 = origins - lbd
            p2 = origins + rfu

            if to_mode == BoxMode.XYZWDH_ABS:
                sizes = p2 - p1
                arr[:, 0:3] = p1 + sizes / 2.
                arr[:, 3:6] = sizes
            else:  # BoxMode.XYZXYZ_ABS
                arr[:, 0:3] = p1
                arr[:, 3:6] = p2

        if single_box:
            return original_type(arr.flatten().tolist()), kwargs
        if is_numpy:
            return arr.numpy(), kwargs
        else:
            return arr, kwargs


class Boxes(object, metaclass=ABCMeta):
    def __init__(self, tensor: torch.Tensor, mode: "BoxMode" = BoxMode.XYZWDH_ABS, **kwargs: Any):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 6)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 6, tensor.size()

        self._tensor = tensor
        self._mode = mode

        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name in ["_fields", "_tensor", "_mode"] or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Boxes!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Boxes of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, mode: "BoxMode", **kwargs: Any) -> "Boxes":
        if mode == BoxMode.XYZWDH_ABS:
            return XYZWDHBoxes(tensor, **kwargs)
        elif mode == BoxMode.XYZXYZ_ABS:
            return XYZXYZBoxes(tensor, **kwargs)
        elif mode == BoxMode.XYZLBDRFU_ABS:
            return XYZLBDRFUBoxes(tensor, **kwargs)
        else:
            print(mode)
            raise NotImplementedError

    def convert(self, to_mode: "BoxMode", **kwargs: Any) -> "Boxes":
        if to_mode == self._mode:
            return self
        tensor, fields = self._get_tensor(to_mode, **kwargs)
        return Boxes.from_tensor(tensor, mode=to_mode, **fields)

    def _get_tensor(self, mode: Optional["BoxMode"] = None, **kwargs: Any) -> Tuple[torch.Tensor, Dict]:
        # TODO:
        is_empty = len(kwargs) == 0
        for k, v in self._fields.items():
            if k not in kwargs:
                kwargs[k] = v
        if mode is None or mode == self._mode:
            assert is_empty, kwargs
            return self._tensor, kwargs
        else:
            return BoxMode.convert(
                self._tensor, from_mode=self._mode, to_mode=mode, **kwargs
            )

    def get_tensor(self, mode: Optional["BoxMode"] = None, **kwargs: Any):
        box = self.convert(self._mode if mode is None else mode, **kwargs)
        return box._tensor

    @property
    def mode(self) -> "BoxMode":
        return self._mode

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        ret = type(self)(self._tensor.clone())
        for k, v in self._fields.items():
            if hasattr(v, "clone"):
                v = v.clone()
            ret.set(k, v)
        return ret

    def to(self, *args: Any, **kwargs: Any) -> "Boxes":
        ret = type(self)(self._tensor.to(*args, **kwargs))
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

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

        ret = cls(b)
        for k, v in self._fields.items():
            ret.set(k, v[item])

        return ret

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

        for k in boxes_list[0]._fields.keys():
            values = [i.get(k) for i in boxes_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            cat_boxes.set(k, values)

        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    def __iter__(self):
        raise NotImplementedError("`Boxes` object is not iterable!")


class XYZXYZBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor, **kwargs: Any):
        """
        Args:
            tensor (Tensor[float]): a Nxk matrix, where k = 6.
            Each row is (x0, y0, z0, x1, y1, z1).
        """
        super().__init__(tensor, mode=BoxMode.XYZXYZ_ABS, **kwargs)

    def get_centers(self) -> torch.Tensor:
        return (self._tensor[:, 0:3] + self._tensor[:, 3:6]) / 2.

    def get_sizes(self) -> torch.Tensor:
        return self._tensor[:, 3:6] - self._tensor[:, 0:3]

    def __repr__(self) -> str:
        return f"XYZXYZBoxes({str(self._tensor)}, fields={self._fields})"


class XYZWDHBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor, **kwargs: Any):
        """
        Args:
            tensor (Tensor[float]): a Nxk matrix, where k = 6.
            Each row is (xc, yc, zc, width, depth, height).
        """
        super().__init__(tensor, mode=BoxMode.XYZWDH_ABS, **kwargs)

    def get_centers(self) -> torch.Tensor:
        return self._tensor[:, 0:3]

    def get_sizes(self) -> torch.Tensor:
        return self._tensor[:, 3:6]

    def __repr__(self) -> str:
        return f"XYZWDHBoxes({str(self._tensor)}, fields={self._fields})"


class XYZLBDRFUBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor, **kwargs: Any):
        """
        Args:
            tensor (Tensor[float]): a Nxk matrix, where k = 6.
            Each row is (xc, yc, zc, width, depth, height).
        """
        assert "origins" in kwargs

        super().__init__(tensor, mode=BoxMode.XYZLBDRFU_ABS, **kwargs)

    def get_centers(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_centers()

    def get_sizes(self) -> torch.Tensor:
        return self.convert(BoxMode.XYZWDH_ABS).get_sizes()

    def __repr__(self) -> str:
        return f"XYZLBDRFUBoxes({str(self._tensor)}, fields={self._fields})"


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

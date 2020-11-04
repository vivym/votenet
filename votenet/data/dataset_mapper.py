import logging
from typing import List, Union

import numpy as np
import torch

from votenet.config import configurable
from votenet.structures import Boxes, BoxMode, Instances

from . import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Votenet Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        use_color: bool,
        use_height: bool,
        num_points: int,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            use_height: whether to use height
        """
        # fmt: off
        self.is_train       = is_train
        self.augmentations  = T.AugmentationList(augmentations)
        self.use_color      = use_color
        self.use_height     = use_height
        self.num_points     = num_points
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        random_x_flip = cfg.INPUT.RANDOM_X_FLIP
        random_y_flip = cfg.INPUT.RANDOM_Y_FLIP
        random_z_rotation = cfg.INPUT.RANDOM_Z_ROTATION
        random_scale = cfg.INPUT.RANDOM_SCALE

        augs = []
        if is_train:
            if random_x_flip > 0:
                augs.append(T.RandomXFlip(prob=random_x_flip))
            if random_y_flip > 0:
                augs.append(T.RandomYFlip(prob=random_y_flip))
            if random_z_rotation is not None:
                augs.append(T.RandomZRotation(angle=random_z_rotation, sample_style="range"))
            if random_scale is not None:
                augs.append(T.RandomScale(scale_factor=random_scale, sample_style="range"))

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "use_color": cfg.INPUT.USE_COLOR,
            "use_height": cfg.INPUT.USE_HEIGHT,
            "num_points": cfg.INPUT.NUM_POINTS,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one scene.

        Returns:
            dict: a format that builtin models in votenet accept
        """
        bbox_mode = dataset_dict["bbox_mode"]
        dataset_dict = np.load(dataset_dict["path"])
        points = dataset_dict["points"]
        point_votes = dataset_dict["point_votes"]
        point_votes_mask = dataset_dict["point_votes_mask"]
        gt_boxes = dataset_dict["gt_boxes"]
        gt_classes = dataset_dict["gt_classes"]

        gt_boxes = BoxMode.convert(gt_boxes, from_mode=bbox_mode, to_mode=BoxMode.XYZWDH_ABS)

        if self.use_color:
            points = points[:, :6]
        else:
            points = points[:, :3]

        # Random point sampling
        choice = np.random.choice(
            points.shape[0], self.num_points, replace=points.shape[0] < self.num_points
        )
        points = points[choice]
        point_votes = point_votes[choice]
        point_votes_mask = point_votes_mask[choice]

        # Apply augmentations
        aug_input = T.AugInput(points, boxes=gt_boxes)
        transforms = self.augmentations(aug_input)
        points, gt_boxes = aug_input.points, aug_input.boxes

        point_votes = transforms.apply_points(point_votes)

        if point_votes.shape[1] == 3:
            point_votes = np.tile(point_votes, (1, 3))

        instances = Instances()
        instances.gt_boxes = Boxes.from_tensor(torch.as_tensor(gt_boxes), mode=BoxMode.XYZWDH_ABS)
        instances.gt_classes = torch.as_tensor(gt_classes)

        if self.use_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points, np.expand_dims(height, 1)], 1)

        return {
            "points": torch.as_tensor(points),
            "point_votes": torch.as_tensor(point_votes),
            "point_votes_mask": torch.as_tensor(point_votes_mask),
            "instances": instances,
        }

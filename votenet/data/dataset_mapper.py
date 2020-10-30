import copy
import logging
from typing import List, Optional, Union

import numpy as np
import torch

from votenet.config import configurable
from votenet.structures import BoxMode, Boxes, Instances

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
        use_height: bool,
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
        self.use_height     = use_height
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "use_height": cfg.INPUT.USE_HEIGHT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one scene.

        Returns:
            dict: a format that builtin models in votenet accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        point_cloud = np.load(dataset_dict["point_cloud"])

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["point_cloud"] = torch.as_tensor(point_cloud)

        instances = Instances()
        if dataset_dict["type"] == "scannet":
            bboxes = np.load(dataset_dict["bboxes"])  # XYZWDH_SID
            instance_labels = np.load(dataset_dict["instance_labels"])
            semantic_labels = np.load(dataset_dict["semantic_labels"])

            size_id = bboxes[:, 6]
            bboxes = BoxMode.convert(bboxes[:, :6], BoxMode.XYZWDH_ABS, BoxMode.XYZXYZ_ABS)
            instances.gt_boxes = bboxes
            instances.gt_size_id = size_id



        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

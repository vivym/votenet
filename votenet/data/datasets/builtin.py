"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .scannet import register_scannet_instances
from .sunrgbd import register_sunrgbd_instances

_PREDEFINED_SPLITS_SCANNET = {
    "scannet_v2": {
        "scannet_v2/train": ("scannet/detection_data", "scannet/meta_data/scannetv2_train.txt"),
        "scannet_v2/val": ("scannet/detection_data", "scannet/meta_data/scannetv2_val.txt"),
        "scannet_v2/test": ("scannet/detection_data", "scannet/meta_data/scannetv2_test.txt"),
    },
}

_PREDEFINED_SPLITS_SUNRGBD = {
    "sunrgbd_v1": {
        "sunrgbd_v1/train": ("sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_train",),
        "sunrgbd_v1/val": ("sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_val",),
    },
    "sunrgbd_v2": {
        "sunrgbd_v2/train": ("sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_train",),
        "sunrgbd_v2/val": ("sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_val",),
    },
}


def register_all_scannet(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_SCANNET.items():
        for key, (data_root, split_file) in splits_per_dataset.items():
            # Assume pre-defined datasets scannet in `./datasets`.
            register_scannet_instances(
                key,
                os.path.join(root, split_file) if "://" not in split_file else split_file,
                os.path.join(root, data_root),
            )


def register_all_sunrgbd(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_SUNRGBD.items():
        for key, (data_root,) in splits_per_dataset.items():
            # Assume pre-defined datasets scannet in `./datasets`.
            register_sunrgbd_instances(
                key,
                os.path.join(root, data_root),
            )


_root = os.getenv("VOTENET_DATASETS", "datasets")
register_all_scannet(_root)
register_all_sunrgbd(_root)

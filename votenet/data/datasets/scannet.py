import os

from fvcore.common.file_io import PathManager

from votenet.structures import BoxMode

from ..catalog import DatasetCatalog, MetadataCatalog


def register_scannet_instances(name, split_file, data_root):
    assert isinstance(name, str), name
    assert isinstance(split_file, (str, os.PathLike)), split_file
    assert isinstance(data_root, (str, os.PathLike)), data_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_scannet(split_file, data_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        split_file=split_file, data_root=data_root, evaluator_type="point_cloud"
    )


def load_scannet(split_file, data_root, name):
    split_file = PathManager.open(split_file).read()

    all_scans = set([
        os.path.basename(file_name)[:12]
        for file_name in os.listdir(data_root)
    ])

    scans = filter(lambda s: len(s) > 0 and s in all_scans, split_file.split("\n"))

    dataset_dicts = [
        {
            "path": os.path.join(data_root, f"{prefix}.npz"),
            "bbox_mode": BoxMode.XYZWDH_ABS,
        }
        for prefix in sorted(scans)
    ]

    return dataset_dicts

import os

from fvcore.common.file_io import PathManager

from votenet.data import DatasetCatalog


def register_scannet_instances(name, split_file, data_root):
    assert isinstance(name, str), name
    assert isinstance(split_file, (str, os.PathLike)), split_file
    assert isinstance(data_root, (str, os.PathLike)), data_root

    DatasetCatalog.register(name, lambda: load_scannet(split_file, data_root, name))


def load_scannet(split_file, data_root, name):
    split_file = PathManager.get_local_path(split_file)

    all_scans = set([
        os.path.basename(file_name)[:12]
        for file_name in os.listdir(data_root)
    ])

    scans = filter(lambda s: len(s) > 0 and s in all_scans, split_file.split("\n"))

    dataset_dicts = [
        {
            "path": os.path.join(data_root, f"{prefix}.npy"),
        }
        for prefix in sorted(scans)
    ]

    return dataset_dicts

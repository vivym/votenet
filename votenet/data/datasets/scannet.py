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

    all_scenes = set([
        os.path.basename(file_name)[:12]
        for file_name in os.listdir(data_root)
    ])

    scenes = filter(lambda s: len(s) > 0 and s in all_scenes, split_file.split("\n"))

    dataset_dicts = [
        {
            "type": "scannet",
            "point_cloud": os.path.join(data_root, f"{prefix}_vert.npy"),
            "bboxes": os.path.join(data_root, f"{prefix}_bbox.npy"),
            "instance_labels": os.path.join(data_root, f"{prefix}_ins_label.npy"),
            "semantic_labels": os.path.join(data_root, f"{prefix}_sem_label.npy"),
        }
        for prefix in sorted(scenes)
    ]

    return dataset_dicts

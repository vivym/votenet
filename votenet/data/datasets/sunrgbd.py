import os

from votenet.data import DatasetCatalog


def register_sunrgbd_instances(name, data_root):
    assert isinstance(name, str), name
    assert isinstance(data_root, (str, os.PathLike)), data_root

    DatasetCatalog.register(name, lambda: load_sunrgbd(data_root, name))


def load_sunrgbd(data_root, name):
    scenes = set([
        os.path.basename(file_name)[:6]
        for file_name in os.listdir(data_root)
    ])

    dataset_dicts = [
        {
            "type": "sunrgbd",
            "point_cloud": os.path.join(data_root, f"{prefix}_pc.npz"),
            "bboxes": os.path.join(data_root, f"{prefix}_bbox.npy"),
            "point_votes": os.path.join(data_root, f"{prefix}_votes.npz"),
        }
        for prefix in sorted(scenes)
    ]

    return dataset_dicts


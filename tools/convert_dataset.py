import os
import sys
from tqdm import tqdm

import numpy as np


def convert_scannet():
    nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(nyu40ids))}

    data_root = "./datasets/scannet/scannet_train_detection_data/"
    new_data_root = "./datasets/scannet/data/"

    scans = set([
        os.path.basename(file_name)[:12]
        for file_name in os.listdir(data_root)
    ])

    for prefix in tqdm(scans):
        points = np.load(f"{data_root}{prefix}_vert.npy")
        instance_bboxes = np.load(f"{data_root}{prefix}_bbox.npy")
        instance_labels = np.load(f"{data_root}{prefix}_ins_label.npy")
        semantic_labels = np.load(f"{data_root}{prefix}_sem_label.npy")

        point_votes = np.zeros([points.shape[0], 3])
        point_votes_mask = np.zeros(points.shape[0])
        for i_instance in np.unique(instance_labels):
            ind = np.where(instance_labels == i_instance)[0]

            if semantic_labels[ind[0]] in nyu40ids:
                x = points[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0

        gt_classes = instance_bboxes[:, 6].astype(np.int64)
        gt_classes[:] = [nyu40id2class[x] for x in gt_classes]
        gt_boxes = instance_bboxes[:, :6]

        np.savez_compressed(
            f"{new_data_root}{prefix}.npz",
            points=points.astype(np.float32),
            point_votes=point_votes.astype(np.float32),
            point_votes_mask=point_votes_mask.astype(np.bool),
            gt_boxes=gt_boxes.astype(np.float32),
            gt_classes=gt_classes.astype(np.int64),
        )


def main():
    dataset = sys.argv[1]
    assert dataset in ["scannet", "sunrgbd"]

    if dataset == "scannet":
        convert_scannet()


if __name__ == '__main__':
    main()

import os
from tqdm import tqdm

import numpy as np


def main():
    data_root = "./datasets/scannet/data/"

    scans = set([file_name for file_name in os.listdir(data_root)])
    count = 0
    for file_name in tqdm(scans):
        dataset = np.load(f"{data_root}{file_name}")
        gt_boxes = dataset["gt_boxes"]

        if gt_boxes.shape[0] == 0:
            print(file_name)
            count += 1

    print(count)


if __name__ == '__main__':
    main()

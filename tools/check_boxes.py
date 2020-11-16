import torch

from votenet.layers import batched_nms_3d


def main():
    obj = torch.load("./pred_boxes.pth", map_location="cpu")
    pred_boxes_i = obj["pred_boxes"]
    scores_i = obj["scores"]

    filter_mask = scores_i > 0.05
    print(filter_mask.sum())
    filter_inds = filter_mask.nonzero(as_tuple=False)
    pred_boxes_i = pred_boxes_i[filter_mask]
    scores_i = scores_i[filter_mask]

    keep = batched_nms_3d(pred_boxes_i, scores_i, filter_inds[:, 1], 0.25)

    print(keep.size())

    filter_inds = filter_inds[keep]
    print(scores_i[keep])
    print(filter_inds[:, 1])


if __name__ == '__main__':
    main()

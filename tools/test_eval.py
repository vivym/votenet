from collections import defaultdict

import numpy as np
import torch

from votenet.structures import BoxMode, Boxes, pairwise_iou


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(all_pred, gt_boxes, ovthresh=0.5, use_07_metric=False):
    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for scan_id, boxes in gt_boxes.items():
        bbox = torch.as_tensor(boxes)
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[scan_id] = {"bbox": bbox, "det": det}
    assert npos > 0

    scan_ids = []
    confidence = []
    BB = []
    for scan_id, pred in all_pred.items():
        for box, score in pred:
            scan_ids.append(scan_id)
            confidence.append(score)
            BB.append(box)
        if scan_id not in class_recs:
            class_recs[scan_id] = {"bbox": torch.as_tensor([]), "det": []}
    confidence = np.array(confidence)
    BB = np.array(BB)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    scan_ids = [scan_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(scan_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        bb = BB[d, :].astype(float)
        R = class_recs[scan_ids[d]]
        BBGT = R["bbox"]
        ovmax = -np.inf
        jmax = -1

        if BBGT.size(0) > 0:
            # compute overlaps
            """
            overlaps = pairwise_iou(
                Boxes.from_tensor(BBGT, mode=BoxMode.XYZWDH_ABS),
                Boxes.from_tensor(bb[None, :], mode=BoxMode.XYZWDH_ABS),
            ).squeeze()

            ovmax, jmax = overlaps.max(dim=0)
            ovmax, jmax = ovmax.item(), jmax.item()
            """
            for j in range(BBGT.shape[0]):
                iou = calc_iou(bb, BBGT[j,...].numpy())
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def eval_predictions(all_pred, all_gt_boxes):
    aps = defaultdict(list)
    # TODO: get class ids from Metadata
    for cls in all_gt_boxes.keys():
        for thresh in range(50, 100, 5):
            rec, prec, ap = voc_eval(
                all_pred[cls],
                all_gt_boxes[cls],
                ovthresh=thresh / 100.0,
                use_07_metric=False,
            )
            aps[thresh].append(ap * 100)

    mAP = {iou: np.mean(x) for iou, x in aps.items()}
    results = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
    print(results)


def main():
    old_pred = torch.load("old_predictions.pth")
    pred_all = old_pred["pred_map_cls"]
    gt_all = old_pred["gt_map_cls"]

    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    eval_predictions(pred, gt)


if __name__ == '__main__':
    main()

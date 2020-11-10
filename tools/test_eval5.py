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
            overlaps = pairwise_iou(
                Boxes.from_tensor(BBGT, mode=BoxMode.XYZWDH_ABS),
                Boxes.from_tensor(bb[None, :], mode=BoxMode.XYZWDH_ABS),
            ).squeeze()

            ovmax, jmax = overlaps.max(dim=0)
            ovmax, jmax = ovmax.item(), jmax.item()

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


def eval_predictions(predictions):
    """
    Evaluate predictions on the given tasks.
    Fill self._results with the metrics of the tasks.
    """
    all_pred = defaultdict(lambda: defaultdict(list))
    all_gt_boxes = defaultdict(lambda: defaultdict(list))
    for prediction in predictions:
        scan_id = prediction["path"]
        topk = 128
        pred_classes = prediction["instances"].pred_classes[:topk].tolist()
        pred_boxes = prediction["instances"].pred_boxes.get_tensor(BoxMode.XYZWDH_ABS)[:topk].tolist()
        scores = prediction["instances"].scores[:topk].tolist()
        gt_classes = prediction["gt_instances"].gt_classes.tolist()
        gt_boxes = prediction["gt_instances"].gt_boxes.get_tensor(BoxMode.XYZWDH_ABS).tolist()

        for cls, box, score in zip(pred_classes, pred_boxes, scores):
            all_pred[cls][scan_id].append((box, score))

        for cls, box in zip(gt_classes, gt_boxes):
            all_gt_boxes[cls][scan_id].append(box)

    aps = defaultdict(list)
    # TODO: get class ids from Metadata
    for cls in all_gt_boxes.keys():
        for thresh in [25, 50]:
            rec, prec, ap = voc_eval(
                all_pred[cls],
                all_gt_boxes[cls],
                ovthresh=thresh / 100.0,
                use_07_metric=False,
            )
            aps[thresh].append(ap * 100)

    mAP = {iou: np.mean(x) for iou, x in aps.items()}
    results = {"AP": np.mean(list(mAP.values())), "AP25": mAP[25], "AP50": mAP[50]}
    print(results)


def main():
    predictions = torch.load("predictions.pth", map_location="cpu")

    eval_predictions(predictions)


if __name__ == '__main__':
    main()

import os
import copy
import logging
import itertools
from collections import OrderedDict, defaultdict

import numpy as np
from fvcore.common.file_io import PathManager
import torch

from votenet.utils import comm
from votenet.data import MetadataCatalog
from votenet.structures import BoxMode, Boxes, pairwise_iou

from .evaluator import DatasetEvaluator


class PointCloudEvaluation(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, distributed, output_dir=None, *, use_fast_impl=True):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        super().__init__()

        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., VoteNet).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"path": input["path"]}

            if "instances" in output:
                prediction["instances"] = output["instances"].to(self._cpu_device)
                prediction["gt_instances"] = input["instances"].to(self._cpu_device)
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[PointCloudEvaluation] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        torch.save(predictions, "predictions.pth")
        all_pred = defaultdict(lambda: defaultdict(list))
        all_gt_boxes = defaultdict(lambda: defaultdict(list))
        for prediction in predictions:
            scan_id = prediction["path"]
            pred_classes = prediction["instances"].pred_classes.tolist()
            pred_boxes = prediction["instances"].pred_boxes.get_tensor(BoxMode.XYZWDH_ABS).tolist()
            scores = prediction["instances"].scores.tolist()
            gt_classes = prediction["gt_instances"].gt_classes.tolist()
            gt_boxes = prediction["gt_instances"].gt_boxes.get_tensor(BoxMode.XYZWDH_ABS).tolist()

            for cls, box, score in zip(pred_classes, pred_boxes, scores):
                all_pred[cls][scan_id].append((box, score))

            for cls, box in zip(gt_classes, gt_boxes):
                all_gt_boxes[cls][scan_id].append(box)
        del predictions

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
        self._results["bbox"] = {"AP": np.mean(list(mAP.values())), "AP25": mAP[25], "AP50": mAP[50]}

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        raise NotImplementedError


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


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

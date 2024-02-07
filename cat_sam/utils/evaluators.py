from typing import Tuple, Dict, Union, List

import torch.nn.functional as F
import numpy as np
import torch
import cat_sam.utils.samhq_misc as misc


def to_numpy(input_tensor: torch.Tensor):
    if hasattr(input_tensor, 'detach'):
        input_tensor = input_tensor.detach()
    if hasattr(input_tensor, 'cpu'):
        input_tensor = input_tensor.cpu()
    return input_tensor.numpy()


class SamHQIoU:

    def __init__(self):
        self.reset()

    @staticmethod
    def compute_iou(preds: torch.Tensor, target: torch.Tensor):
        assert target.shape[1] == 1, 'only support one mask per image now'
        if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
            postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
        else:
            postprocess_preds = preds
        iou_list = []
        for i in range(0, len(preds)):
            iou_list.append(misc.mask_iou(postprocess_preds[i], target[i]))
        return iou_list


    @staticmethod
    def compute_boundary_iou(preds: torch.Tensor, target: torch.Tensor):
        assert target.shape[1] == 1, 'only support one mask per image now'
        if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
            postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
        else:
            postprocess_preds = preds
        biou_list = []
        for i in range(0, len(preds)):
            biou_list.append(misc.boundary_iou(target[i], postprocess_preds[i]))
        return biou_list


    def update(self, label_trues: List[torch.Tensor], label_preds: List[torch.Tensor], index_name: List):
        assert len(label_preds) == len(label_trues)

        for i in range(len(label_trues)):
            if index_name[i] not in self.index_results.keys():
                self.index_results[index_name[i]] = {}

            if label_trues[i].max() <= 1.0:
                label_trues[i] *= 255.0

            curr_iou = self.compute_iou(label_preds[i], label_trues[i])[0]
            if isinstance(curr_iou, torch.Tensor):
                curr_iou = curr_iou.item()
            self.index_results[index_name[i]]['iou'] = curr_iou

            curr_biou = self.compute_boundary_iou(label_preds[i], label_trues[i])[0]
            if isinstance(curr_biou, torch.Tensor):
                curr_biou = curr_biou.item()
            self.index_results[index_name[i]]['biou'] = curr_biou


    def compute(self) -> Tuple[Dict, Dict]:
        results_dict = {
            "Mean Foreground IoU": sum([item['iou'] for item in self.index_results.values()]) / len(self.index_results),
            "Mean Foreground BIoU": sum([item['biou'] for item in self.index_results.values()]) / len(self.index_results),
        }
        return results_dict, self.index_results


    def reset(self):
        self.index_results = {}


class StreamSegMetrics:
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.reset()

    def update(self,
               label_trues: Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
               label_preds: Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
               index_name: List):
        for masks in [label_preds, label_trues]:
            for i in range(len(masks)):
                if not isinstance(masks[i], np.ndarray):
                    masks[i] = to_numpy(masks[i])
                if len(masks[i].shape) == 2:
                    masks[i] = masks[i][None, None, :]
                if len(masks[i].shape) == 3:
                    masks[i] = masks[i][None, :]
                if len(masks[i].shape) != 4:
                    raise RuntimeError

        for i, (lt, lp) in enumerate(zip(label_trues, label_preds)):
            index_hist = self._fast_hist(lt.flatten(), lp.flatten())
            self.confusion_matrix += index_hist
            self.index_results[index_name[i]] = self.compute(hist=index_hist)[0]


    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def compute(self, hist=None) -> Tuple[Dict, Dict]:
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        if hist is None:
            hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        results_dict = {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Mean Foreground IoU": np.nanmean(iu[1:]) if len(iu) > 1 else 0.0
        }
        for i, class_name in enumerate(self.class_names):
            results_dict[f'{class_name} IoU'] = iu[i]

        return results_dict, self.index_results

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.index_results = {}

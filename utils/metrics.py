import cv2
import ray
import time
import torch

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn.cluster import DBSCAN, OPTICS
from torchmetrics import JaccardIndex as JI
from torchmetrics import ConfusionMatrix as CM
from scipy.spatial import ConvexHull


# pytorch metrics
class PositiveIoUScore(nn.Module):
    __name__ = 'iou_score'

    def __init__(self):
        super(PositiveIoUScore, self).__init__()
        self.metric = JI(num_classes=2, absent_score=1, reduction='none').cuda()

    def forward(self, inputs, targets):
        ious = self.metric(inputs, targets.int())
        return ious[1]


class NegativeIoUScore(nn.Module):
    __name__ = 'bg_iou'

    def __init__(self):
        super(NegativeIoUScore, self).__init__()
        self.metric = JI(num_classes=2, reduction='none').cuda()

    def forward(self, inputs, targets):
        ious = self.metric(inputs, targets.int())
        return ious[0]


class TrueNegativeRate(nn.Module):
    __name__ = 'tnr'

    def __init__(self, normalize=True):
        super(TrueNegativeRate, self).__init__()
        self.metric = CM(num_classes=2, normalize='true').cuda()
    
    def forward(self, inputs, targets):
        cmat = self.metric(inputs, targets.int())
        return cmat[0][0]


class FalsePositiveRate(nn.Module):
    __name__ = 'fpr'

    def __init__(self, normalize=True):
        super(FalsePositiveRate, self).__init__()
        self.metric = CM(num_classes=2, normalize='true').cuda()
    
    def forward(self, inputs, targets):
        cmat = self.metric(inputs, targets.int())
        return cmat[0][1]


class FalseNegativeRate(nn.Module):
    __name__ = 'fnr'

    def __init__(self, normalize=True):
        super(FalseNegativeRate, self).__init__()
        self.metric = CM(num_classes=2, normalize='true').cuda()
    
    def forward(self, inputs, targets):
        cmat = self.metric(inputs, targets.int())
        return cmat[1][0]


class TruePositiveRate(nn.Module):
    __name__ = 'tpr'

    def __init__(self, normalize=True):
        super(TruePositiveRate, self).__init__()
        self.metric = CM(num_classes=2, normalize='true').cuda()
    
    def forward(self, inputs, targets):
        cmat = self.metric(inputs, targets.int())
        return cmat[1][1]


# custom metrics
@ray.remote
def remote_to_blobs(mask, threshold=.5, blobber=DBSCAN):
    """"""
    canvas = np.zeros(mask.shape)
    contours = []
    
    # get all positive prediction coordinates
    coords = np.flip(np.column_stack(np.where(mask > threshold)), axis=1)
    
    if len(coords) > 10:
        # use clustering algorithm to find labels per pixel coordinate
        clustering = blobber(eps=5, min_samples=10).fit(coords)
        coord_labels = clustering.labels_
        
        # get non noisy cluster labels
        labels = np.unique(coord_labels)
        labels = labels[labels >= 0]
        
        for label in labels:
            cluster = coords[coord_labels == label]
            contour = cluster[ConvexHull(cluster).vertices]
            
            contours.append(contour)
    
        canvas = cv2.drawContours(canvas, contours, -1, 1, -1)
    
    return canvas


@ray.remote
def remote_calculate(preds, target, blobbing=True):
    """"""
    if blobbing:
        preds = to_blobs(preds)
        target = to_blobs(target)

    union = target + preds

    # change background value of prediction for efficient overlap computation
    preds[preds == 0] = -1

    # calculate blob overlap
    overlap = target - preds

    area_of_overlap = np.count_nonzero(overlap == 0)
    area_of_union = np.count_nonzero(union > 0)

    if area_of_union  > 0:
        blobs_iou = area_of_overlap / area_of_union
    elif area_of_overlap == 0 and area_of_union == 0:
        blobs_iou = 1
    else:
        blobs_iou = 0

    return blobs_iou


def to_blobs(mask, threshold=.5, blobber=DBSCAN, eps=5):
    """"""
    canvas = np.zeros(mask.shape)
    contours = []
    
    # get all positive prediction coordinates
    coords = np.flip(np.column_stack(np.where(mask > threshold)), axis=1)
    
    if len(coords) > 0:
        # use clustering algorithm to find labels per pixel coordinate
        clustering = blobber(eps=eps, min_samples=10).fit(coords)
        coord_labels = clustering.labels_
        
        # get non noisy cluster labels
        labels = np.unique(coord_labels)
        labels = labels[labels >= 0]
        
        for label in labels:
            cluster = coords[coord_labels == label]
            try:
                contour = cluster[ConvexHull(cluster).vertices]
            except:
                return canvas, False
            
            contours.append(contour)
    
        canvas = cv2.drawContours(canvas, contours, -1, 1, -1)
    
    return canvas, True


def calculate(preds, target, blobbing=True):
    """"""
    if blobbing:
        preds, pred_success = to_blobs(preds)
        if not pred_success:
            return 0, False
            
        target, target_success = to_blobs(target)
        if not target_success:
            return 0, False

    union = target + preds

    # change background value of prediction for efficient overlap computation
    preds[preds == 0] = -1

    # calculate blob overlap
    overlap = target - preds

    area_of_overlap = np.count_nonzero(overlap == 0)
    area_of_union = np.count_nonzero(union > 0)

    if area_of_union > 0:
        blobs_iou = area_of_overlap / area_of_union
    elif area_of_overlap == 0 and area_of_union == 0:
        blobs_iou = 1
    else:
        blobs_iou = 0

    return blobs_iou, True


def unravel(batches):
    """"""
    unraveled = []
   
    for batch in batches:
        if isinstance(batch, torch.Tensor):
            batch = batch.squeeze(1).cpu().numpy()
        else:
            batch = batch.squeeze(1)

        for sample in batch:
            unraveled.append(sample)

    return unraveled


class BlobOverlap():
    __name__ = 'blob_overlap'

    def __init__(self, num_workers=1):
        self.score = 0
        self.count = 0
        
        if num_workers > 1:
            ray.init(num_cpus=num_workers)

        self.num_workers = num_workers


    def update(self, preds, targets):
        """"""
        # unravel all predictions and targets
        preds = np.array(unravel(preds))
        targets = np.array(unravel(targets))
        
        score = 0
        
        for i, (pred, target) in enumerate(zip(preds, targets)):
            iou, success = calculate(pred, target)

            score = iou
            
            self.score += iou
            self.count += 1 if success else 0

        # # torch to numpy
        # if isinstance(preds, torch.Tensor):
        #     preds = preds.squeeze(1).cpu().numpy()
        #     targets = targets.squeeze(1).cpu().numpy()
        
        # # calculate per batch
        # if self.num_workers > 1:
        #     pids = []
        #     for i, (pred, target) in enumerate(zip(preds, targets)):
        #         pids.append(remote_calculate.remote(pred, target))

        #     score = sum(ray.get(pids))
        # else:
        #     score = 0
        #     for i, (pred, target) in enumerate(zip(preds, targets)):
        #         score += calculate(pred, target)
            
        # score /= (i + 1)
        
        # self.score += score
        # self.count += 1

        return score

    
    def update_all(self, all_preds, all_targets):
        """ performs some scheduling to optimize 
        wall time while using multiple workers """
        
        # unravel all predictions and targets
        all_preds = unravel(all_preds)
        all_targets = unravel(all_targets)

        # combine and sort in ascending positive sample count
        all_samples = np.array(all_preds + all_targets)
        px_counts = np.array([np.count_nonzero(sample > 0) for sample in all_samples])
        order = np.argsort(px_counts)

        # parallel blobbing
        pids = []
        for idx in order:
            pids.append(remote_to_blobs.remote(all_samples[idx]))

        # restore original order and aggregate results
        restored = list(np.array(pids)[np.argsort(order)])
        all_blobs = ray.get(restored)

        # calculate mean score
        n = len(all_preds)
        scores = [calculate(all_blobs[i].copy(), all_blobs[i + n].copy(), blobbing=False) for i in range(n)]

        self.score += sum(scores)
        self.count += n

        return sum(scores)

        
    def compute(self):
        """"""
        return self.score / self.count


# everything below sourced from: segmentation-models-pytorch:
# github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/meter.py

class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
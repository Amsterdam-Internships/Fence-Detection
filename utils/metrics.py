import numpy as np
import torch.nn as nn

from torchmetrics import JaccardIndex


class PositiveIoUScore(nn.Module):
    __name__ = 'iou_score'

    def __init__(self):
        super(PositiveIoUScore, self).__init__()
        self.metric = JaccardIndex(num_classes=2, reduction='none').cuda()

    def forward(self, inputs, targets):
        ious = self.metric(inputs, targets.int())
        return ious[1]


class NegativeIoUScore(nn.Module):
    __name__ = 'bg_iou'

    def __init__(self):
        super(NegativeIoUScore, self).__init__()
        self.metric = JaccardIndex(num_classes=2, reduction='none').cuda()

    def forward(self, inputs, targets):
        ious = self.metric(inputs, targets.int())
        return ious[0]


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
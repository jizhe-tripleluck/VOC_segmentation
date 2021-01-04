import torch.nn as nn
import torch.tensor as Tensor
from training.utils import batch_labels_one_hot

SMOOTH = 1e-6


class MIoU(nn.Module):
    def __init__(self):
        super(MIoU, self).__init__()

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        labels = batch_labels_one_hot(labels)

        intersection = (outputs & labels).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((2, 3))         # Will be zero if both are 0

        IoU = (union > 0.5) * (intersection + SMOOTH) / (union + SMOOTH)  # Smooth our devision to avoid 0/0
        mIoU = IoU.mean(1)     # Average across all the classes

        return mIoU.sum()  # Or mIoU.mean() if you are interested in average across the batch

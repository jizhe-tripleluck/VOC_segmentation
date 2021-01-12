import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F
from training.utils import batch_labels_one_hot

SMOOTH = 1e-6


class OldMIoU(nn.Module):
    def __init__(self):
        super(OldMIoU, self).__init__()

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        labels = batch_labels_one_hot(labels)

        intersection = (outputs & labels).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((2, 3))         # Will be zero if both are 0

        IoU = intersection / (union + SMOOTH)  # Smooth our devision to avoid 0/0
        mIoU = IoU.sum(1) / (IoU > 0).sum(1)     # Average across all the classes

        return mIoU.mean()  # Average across the batch


class MIoU(nn.Module):
    def __init__(self):
        super(MIoU, self).__init__()

    def forward(self, logits: Tensor, true: Tensor, eps: float=1e-7) -> Tensor:
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, logits.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        classes_iou = intersection / (union + eps)
        jacc_loss = classes_iou.sum() / (classes_iou > 0).sum()
        return jacc_loss

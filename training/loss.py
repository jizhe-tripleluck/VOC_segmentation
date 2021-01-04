import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F

SMOOTH = 1e-6


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.softmax = nn.Softmax2d()

    def forward(self, outputs: Tensor, labels: Tensor, eps: float=1e-06) -> Tensor:
        # outputs = outputs[:, 1:, :, :]  # Possibly ignore background class
        # labels = labels[:, 1:, :, :]

        outputs = self.softmax(outputs)  # Softmax to get probs

        axes = tuple(range(2, len(outputs.shape)))  # H and W are the sum axises
        numerator = 2.0 * torch.sum(outputs * labels, axes)  # Estimate intersection
        denominator = torch.sum(outputs + labels, axes)  # Estimate sum of cardinalities

        return (1.0 - torch.mean((numerator + eps) / (denominator + eps), 1)).sum()  # Compute soft Dice score


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

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
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return 1.0 - jacc_loss

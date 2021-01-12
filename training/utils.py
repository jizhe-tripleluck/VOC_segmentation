"""One-hot encoding utils"""

import torch
import torch.tensor as Tensor


def make_one_hot(outputs: Tensor, num_classes: int=21) -> Tensor:
    """Get one-hot encoded version of tensor.
        Args:
            outputs: tensor with shape [ B x 1 x H x W ]
            num_classes: total number of classes in the dataset
        Returns:
            one_hot_outputs: tensor with shape [ B x C x H x W ]"""
    one_hot = torch.zeros((outputs.shape[0], num_classes,
                           outputs.shape[2], outputs.shape[3]),
                          dtype=torch.int, requires_grad=False).cuda()
    return one_hot.scatter_(1, outputs.data, 1)


def batch_labels_one_hot(labels: Tensor, num_classes: int=21) -> Tensor:
    """Get one-hot encoded version of tensor.
        Args:
            labels: tensor with shape [ B x H x W ]
            num_classes: total number of classes in the dataset
        Returns:
            one_hot_labels: tensor with shape [ B x C x H x W ]"""
    return make_one_hot(labels.unsqueeze(1), num_classes)


def labels_one_hot(labels: Tensor, num_classes: int=21) -> Tensor:
    """Get one-hot encoded version of tensor.
        Args:
            labels: tensor with shape [ H x W ]
            num_classes: total number of classes in the dataset
        Returns:
            one_hot_labels: tensor with shape [ C x H x W ]"""
    one_hot = torch.zeros((num_classes, labels.shape[0],
                           labels.shape[1]), dtype=torch.int, requires_grad=False).cuda()
    return one_hot.scatter_(0, labels.unsqueeze(0).data, 1)


def discrete_softmax(outputs: Tensor) -> Tensor:
    """Set all class maximimums to 1, other elements to 0
        Args:
            outputs: tensor with shape [ B x C x H x W ]
        Returns:
            one_hot_outputs: tensor with shape [ B x C x H x W ] in one-hot format"""
    _, argmax = torch.max(outputs.data, 1, keepdim=True)
    return make_one_hot(argmax, outputs.shape[1])

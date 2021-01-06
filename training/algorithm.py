import torch.optim as optim
import torch.nn as nn
from training.loss import JaccardLoss as Loss
from training.metric import MIoU as Metric


def get_loss() -> nn.Module:
    """Get current loss.
        Args:

        Returns:
            loss: an instanse of loss"""
    return Loss()


def get_metric() -> nn.Module:
    """Get current metric.
        Args:

        Returns:
            metric: an instanse of metric"""
    return Metric()


def get_optimizer(net: nn.Module, params: list=[]) -> optim.Optimizer:
    """Get optimizer with specified initial params.
        Args:
            net: an instance of model
            params: list of initial params
        Returns:
            optim: optimizer"""
    # return optim.SGD(net.parameters(), lr=params[0], momentum=params[1])
    # return optim.RMSprop(net.parameters(), lr=0.000002, weight_decay=1e-8, momentum=0.9)
    return optim.SGD(net.parameters(), lr=0.001)


def update_optimizer(optimizer: optim.Optimizer, params: list) -> None:
    """Update optimizer params.
        Args:
            optimizer: current optimizer
            params: list of new params
        Returns: """
    for g in optimizer.param_groups:
        g['lr'] = params[0]
        g['momentum'] = params[1]
    pass

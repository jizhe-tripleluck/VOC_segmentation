import torch
import torch.nn as nn
import training.dataset as ds
import training.algorithm as algorithm
import training.utils as utils


def test(net: nn.Module) -> float:
    metric = algorithm.get_metric()
    test_accuracy = 0.0
    total = len(ds.testloader.dataset)
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)

            test_accuracy += metric(outputs, labels).item()
    return round(100 * test_accuracy / total, 2)

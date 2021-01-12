import torch
import torch.nn as nn
import training.dataset as ds
import training.algorithm as algorithm
# import training.utils as utils


def eval(net: nn.Module) -> (float, float):
    criterion = algorithm.get_loss()  # Create an instance of loss
    metric = algorithm.get_metric()
    average_loss = 0.0
    test_accuracy = 0.0

    total = len(ds.trainset)
    with torch.no_grad():
        for data in ds.trainloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            # ds.imshow(images[0, :, :, :])
            # ds.labshow(labels[0, :, :])
            # ds.labshow(torch.max(outputs, 1)[1][0, :, :])
            input()
            average_loss += criterion(outputs, labels).item() * outputs.shape[0]
            test_accuracy += metric(outputs, labels).item() * outputs.shape[0]
    return 100.0 * test_accuracy / total, average_loss / total  # compute metric from loss

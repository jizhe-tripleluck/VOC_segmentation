import torch
import torch.nn as nn
import training.dataset as ds
import training.algorithm as algorithm
# import training.utils as utils


def eval(net: nn.Module) -> (float, float):
    criterion = algorithm.get_loss()  # Create an instance of loss
    # metric = algorithm.get_metric()
    average_loss = 0.0
    # test_accuracy = float("NaN")

    total = len(ds.testset) // 10     # Test only on the small fraction of the data
    current = 0
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            average_loss += criterion(outputs, labels).item() * outputs.shape[0]
            # predicted = utils.discrete_softmax(outputs)
            # test_accuracy += metric(predicted, labels).item()
            current += outputs.shape[0]
            if current >= total:
                break
    average_loss /= total  # Our loss and metric are bound together, so we only need loss
    return 100 * (1.0 - average_loss), average_loss  # compute metric from loss

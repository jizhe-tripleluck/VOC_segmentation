import torch
import torch.nn as nn
import training.dataset as ds
import training.algorithm as algorithm
import training.scheduler as scheduler
import training.validation as validation
# import training.utils as utils
import log_utils.log_tensorboard as log
from progress.bar import IncrementalBar


# Train model for 'epoch_count' epochs
def train(net: nn.Module, epoch_count: int, start_epoch: int=0,
          use_scheduler: bool=False) -> None:
    criterion = algorithm.get_loss()  # Create loss object
    if use_scheduler:                 # Create optimizer
        optimizer = algorithm.get_optimizer(net,
                                            scheduler.params_list[start_epoch])
    else:
        optimizer = algorithm.get_optimizer(net)
    # metric = algorithm.get_metric()

    total = len(ds.trainset)          # Total number of imgs in dataset
    bar_step = total // 50            # Progressbar step

    best_acc = 0.0

    for epoch_idx in range(start_epoch, epoch_count):
        net.train()

        if use_scheduler and epoch_idx > 0:  # Update lr and other params if needed
            algorithm.update_optimizer(optimizer,
                                       scheduler.params_list[epoch_idx])

        # Set init values to zero
        average_loss = 0.0
        # train_accuracy = float("NaN")
        curr_iter = 0

        # Progressbar
        iter_bar = IncrementalBar("Current progress", max=total,
                                  suffix='%(percent)d%%')
        
        for _, data in enumerate(ds.trainloader, 0):
            # Compute forward
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)

            # Stats (old)
            # predicted = utils.discrete_softmax(outputs)
            # train_accuracy += (metric(predicted, labels)).item()

            # Compute loss and backward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Add batch loss to get average after epoch is finished
            average_loss += loss.item() * outputs.shape[0]

            # Progressbar things
            if curr_iter >= bar_step:
                iter_bar.next(bar_step)
                curr_iter -= bar_step
            curr_iter += ds.batch_size

        iter_bar.goto(total)
        iter_bar.finish()

        # Compute avg train loss and accuracy
        average_loss /= total
        train_accuracy = 100.0 * (1.0 - average_loss)  # train_accuracy / total

        # Compute avg test loss and accuracy
        net.eval()
        test_accuracy, test_loss = validation.eval(net)

        # Add to log
        # log.add(epoch_idx, (train_accuracy, test_accuracy,
        #                     average_loss, test_loss))

        # Flush log changes
        # log.save()

        # Print useful numbers
        print('[%d, %5d] average loss: %.3f, test loss: %.3f' %
              (epoch_idx, total, average_loss, test_loss))
        print('Train accuracy: %.2f %%' % train_accuracy)
        print('Test accuracy: %.2f %%' % test_accuracy)

        # Save model if it scored better than previous
        if test_accuracy > best_acc:
            PATH = 'model_instances/net_tmp_epoch_%d_acc_%.2f%%.pth' % (epoch_idx, test_accuracy)
            torch.save(net.state_dict(), PATH)
            best_acc = test_accuracy
    # End of training
    print('Complete')

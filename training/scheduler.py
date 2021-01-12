# Optimizer params for each epoch

params_list = [[0.001]] * 50 + [[0.0001]] * 20 +\
              [[0.00001]] * 20 + [[0.000001]] * 10


def count_epoch() -> int:
    """Count total epochs (when scheduler is turned on)
        Returns:
            epoch_count: total number of epochs"""
    return len(params_list)

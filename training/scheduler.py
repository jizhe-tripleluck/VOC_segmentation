# Optimizer params for each epoch

# params_list = [[0.001, 0.9] for _ in range(2)] +\
#               [[0.0003, 0.9] for _ in range(2)] +\
#               [[0.00008, 0.9] for _ in range(2)] +\
#               [[0.000006, 0.9] for _ in range(2)]

params_list = [[0.000001, 0.9]]


def count_epoch() -> int:
    """Count total epochs (when scheduler is turned on)
        Returns:
            epoch_count: total number of epochs"""
    return len(params_list)

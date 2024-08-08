import os
import random
import time

import numpy as np
import torch


def set_seed(seed=42, cudnn_deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def timer(func):
    def wrapper(*args, **kwargs):
        config = kwargs["config"]
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"\nElapsed time: {elapsed_time:.2f} seconds",
            file=open(config.log_path, "a"),
        )
        return result

    return wrapper

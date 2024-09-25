import logging

import torch


logger = logging.getLogger()


def check_and_return_device(device: str):

    if device in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.info("CUDA device not available, reverting to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device

import logging
import subprocess as sp
import os
import numpy as np
from datetime import datetime, time


def create_logger(file_loc, log_name="spike_logger"):
    # create logger
    logger = logging.getLogger(log_name)
    print(logger)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    # create file handler which logs even debug messages
    fh = logging.FileHandler(file_loc)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def get_gpu_memory():
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def choose_gpu(gpu_to_use=None):
    if gpu_to_use is None:
        free_gpu_memory = get_gpu_memory()
        print("Free memory: {}".format(free_gpu_memory))
        gpu_to_use = np.argmax(free_gpu_memory)
    print("GPU to use: {}".format(gpu_to_use))
    print("Setting GPU")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_to_use)


def is_time_between(time_range):
    now = datetime.now().strftime("%H:%M")
    if time_range[1] < time_range[0]:
        return now >= time_range[0] or now <= time_range[1]
    return time_range[0] <= now <= time_range[1]

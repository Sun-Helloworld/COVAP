import threading
from typing import Any, Callable, List
import time
import torch
import torch.distributed as dist
from queue import Queue
import os

import struct
import fcntl
import socket
import psutil
import subprocess
import math
import numpy as np

from sklearn.cluster import SpectralClustering


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    "Averages the input gradient tensor by allreduce and returns a future."
    fut = dist.all_reduce(tensor, group=group_to_use,
                          async_op=True).get_future()

    def div_by_group_size(fut):
        ret = [fut.value()[0].div_(group_to_use.size())]
        return ret

    return fut.then(div_by_group_size)

def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future:
    """
    This DDP communication hook just calls ``allreduce`` using ``GradBucket``
    tensors. Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result. If user registers this hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    #print("bucket len:", bucket.get_tensor().nelement() * bucket.get_tensor().element_size(), "\ttype:", bucket.get_tensor().dtype)
    
    global layers,bucket_number,key_ef
    buctensor = bucket.get_tensor()

    if not bucket_number == bucket.get_bucket_number():
        bucket_number = bucket.get_bucket_number()
    
    return _allreduce_fut(process_group, bucket.get_tensor())

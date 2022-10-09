# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')

args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank],bucket_cap_mb=15)
    import ddp_comm_hooks_new
    ddp_model.register_comm_hook(state=None, hook=ddp_comm_hooks_new.default_hooks.allreduce_hook)
    #ddp_model.register_comm_hook(dist.group.WORLD, ddp_comm_hooks_new.default_hooks.fp16_compress_hook)
    # state = ddp_comm_hooks_new.powerSGD_hook.PowerSGDState(process_group=dist.group.WORLD, matrix_approximation_rank=1,
    #                               start_powerSGD_iter=10, min_compression_rate=0.5)
    # ddp_model.register_comm_hook(state, ddp_comm_hooks_new.powerSGD_hook.powerSGD_hook)
    config.num_epochs = 5

    train(config, ddp_model, train_iter, dev_iter, test_iter)

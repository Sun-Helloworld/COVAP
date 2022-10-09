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
    global key_ef,bucket_number
    #if key_ef < bucket_number:
    key_ef+=1
    def div_by_group_size(fut):
        ret = [fut.value()[0].div_(group_to_use.size())]
        return ret

    return fut.then(div_by_group_size)

layers = {}
_local_threshold = {}
_global_threshold = {}
allreduce_counter = {}
_boundaries={}
_region_offsets = {}

dsts = None
srcs = None
key_ef = 0
bucket_number = -1
num_step = 0

def list_rotate(l, n):
    return l[-n:] + l[:-n]

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
    global layers,bucket_number,key_ef,allreduce_counter,_local_threshold,_global_threshold,_boundaries,_region_offsets,dsts,srcs
    buctensor = bucket.get_tensor()

    if not bucket_number == bucket.get_bucket_number():
        bucket_number = bucket.get_bucket_number()
    
    if key_ef < bucket_number:
        print(bucket.get_index(), bucket.get_tensor().numel())

    # if key_ef >= bucket_number:
    #     if layers.get(str(bucket.get_index())) == None:
    #         layers[str(bucket.get_index())] = torch.zeros(bucket.get_tensor().size()).to(buctensor.device)
    #         _local_threshold[str(bucket.get_index())] = 0
    #         _global_threshold[str(bucket.get_index())] = 0
    #         allreduce_counter[str(bucket.get_index())] = 0
    #         _boundaries[str(bucket.get_index())] = dist.group.WORLD.size() * [0]
    #         _region_offsets[str(bucket.get_index())] = dist.group.WORLD.size() * [0]
    #         dsts = list(range(dist.group.WORLD.size()))
    #         srcs = dsts[::-1]
    #         dsts = list_rotate(dsts, -dist.get_rank())
    #         srcs = list_rotate(srcs, dist.get_rank()+1)
    #     #buctensor = bucket.get_tensor() + layers[str(bucket.get_index())].to(bucket.get_tensor().device)
    #     return oktopk(process_group, buctensor, bucket.get_index())
    # else:
    #     return _allreduce_fut(process_group, bucket.get_tensor())
    
    # if key_ef >= bucket_number:
    #     if layers.get(str(bucket.get_index())) == None:
    #         layers[str(bucket.get_index())] = torch.zeros(bucket.get_tensor().size()).to(buctensor.device)
    #     buctensor = bucket.get_tensor() + layers[str(bucket.get_index())].to(bucket.get_tensor().device)
    # return _allgather_topk(process_group, buctensor, bucket.get_index())
    # return _allgather_dgc(process_group, buctensor, bucket.get_index())
    # return _allgather_randomk(process_group, buctensor, bucket.get_index())
    # return _allgather_efsignsgd(process_group, buctensor, bucket.get_index())
    
    return _allreduce_fut(process_group, bucket.get_tensor()) # if you want to use OGC, compile pytorch and use this method, same as baseline of Pytorch DDP.
    

def ratio2threshold(tensor, bucindex=None, ratio=0.05):
    global layers
    with torch.no_grad():
        numel = tensor.numel()
        k = max(int(numel * ratio), 1)
        tensor += layers[str(bucindex)].to(tensor.device)
        values, indexes = torch.topk(torch.abs(tensor), k=k)
        layers[str(bucindex)] = tensor

    return float(values[values.numel()-1].item())

def add2residual(tensor=None, bucindex=None, thrd=None, tk=None):
    global layers
    with torch.no_grad():
        tensor += layers[str(bucindex)].to(tensor.device)
        layers[str(bucindex)] = tensor

        abs_tensor = torch.abs(tensor)
        loops = 0
        thres = thrd
        while loops < 5:
            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            if indexes.numel() > 4*tk//3:
                thres *= 1.03
            else:
                break
            loops += 1

    return thres

def compressbythresholdlong(tensor, thres=0.0):
    with torch.no_grad():
        abs_tensor = torch.abs(tensor)

        one_indexes = abs_tensor > thres
        indexes = one_indexes.nonzero().data.squeeze().view(-1)
        return indexes

def compressbythreshold(tensor, thres=0.0):
    with torch.no_grad():
        abs_tensor = torch.abs(tensor)

        one_indexes = abs_tensor > thres
        indexes = one_indexes.nonzero().data.squeeze().view(-1)
        values = tensor.data[indexes]

        indexes = indexes.type(torch.IntTensor)
        return indexes, values

def k2globalthreshold(tensor, k=0):
    numel = tensor.numel()
    kk = min(numel, k)
    with torch.no_grad():
        values, indexes = torch.topk(torch.abs(tensor.data), k=kk)
        global_threshold = float(values[values.numel()-1].item())
        values = tensor[indexes]
        #indexes = indexes.type(torch.IntTensor)
    return values, indexes, global_threshold

# this implementation is referred to https://github.com/Shigangli/Ok-Topk, but we re-implemented it using NCCL
def oktopk(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int) -> torch.futures.Future:
    global bucket_number,layers,bucket_number,key_ef,allreduce_counter,_local_threshold,_global_threshold,_boundaries,_region_offsets,dsts,srcs
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    cstime = time.time()
    local_threshold_recompute_interval = 32
    global_threshold_recompute_interval = 32
    region_repartition_interval = 64
    scale = 1.012
    density = 0.02
    tensor_size = tensor.numel()
    topk_value = int(tensor_size * density)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = group_to_use.size()
    new_name = str(bucindex)
    rank = dist.get_rank()

    if allreduce_counter[new_name] % local_threshold_recompute_interval == 0:
        _local_threshold[new_name] = ratio2threshold(tensor=tensor, bucindex=bucindex, ratio=density)
    else:
        _local_threshold[new_name] = add2residual(tensor=tensor, bucindex=bucindex, thrd=_local_threshold[new_name], tk=topk_value)
    
    local_threshold = _local_threshold[new_name]

    # region repartition
    if allreduce_counter[new_name] % region_repartition_interval == 0:
        with torch.no_grad():
            indexes = compressbythresholdlong(tensor=tensor, thres=local_threshold)
            indexes = indexes.type(torch.LongTensor)
            local_topk_indexes = indexes.cpu().numpy()

        me1 = torch.cuda.Event()
        torch.cuda.current_stream().record_event(me1)

        index_chunk = local_topk_indexes.size // num_workers
        index_boundaries = np.zeros(num_workers, dtype='int64')
        for i in range(num_workers):
            index_boundaries[i] = index_chunk * i
        region_boundaries = local_topk_indexes[index_boundaries[1:]]

        global_boundaries=torch.tensor(region_boundaries.tolist(),dtype=torch.int64,device=tensor.device)
        
        dist.reduce(global_boundaries,dst=0, group=group_to_use)

        dist.broadcast(global_boundaries, 0, group=group_to_use)

        global_boundaries = global_boundaries.cpu().numpy()

        global_boundaries //= num_workers

        for i in range(num_workers):
            if i == 0:
                _boundaries[new_name][i] = global_boundaries[i]
            elif i == num_workers-1:
                _boundaries[new_name][i] = tensor_size-global_boundaries[i-1]
            else:
                _boundaries[new_name][i] = global_boundaries[i]-global_boundaries[i-1]

        assert sum(_boundaries[new_name]) == tensor_size

        for i in range(num_workers):
            if i == 0:
                _region_offsets[new_name][i] = 0
            else:
                _region_offsets[new_name][i] = global_boundaries[i-1]

    boundaries = _boundaries[new_name]

    region_offsets = _region_offsets[new_name]

    with torch.no_grad():
        split_tensors = torch.split(tensor, boundaries)

    assert len(split_tensors) == num_workers
    reduced_t = torch.zeros_like(split_tensors[rank])

    throttle = min(4, num_workers)
    msg_chunks = math.ceil(num_workers/throttle)
    ssizes = torch.zeros(num_workers,dtype=torch.int32,device=tensor.device)
    rsizes = torch.zeros(num_workers,dtype=torch.int32,device=tensor.device)
    r_offsets = np.zeros(num_workers, dtype='int32')

    all_value_sbuffers = []
    all_index_sbuffers = []
    split_topk_indexes = []
    with torch.no_grad():
        for i in range(num_workers):
            indexes, values = compressbythreshold(tensor=split_tensors[i], thres=local_threshold)
            ssizes[i] = torch.numel(values)
            send_index_buffer = indexes.cpu().numpy().astype(np.int32)
            send_value_buffer = values.cpu().numpy().astype(np.float32)
            all_index_sbuffers.append(send_index_buffer)
            all_value_sbuffers.append(send_value_buffer)
            findexes = indexes.cpu().numpy() + region_offsets[i]
            split_topk_indexes.append(findexes)

    # transpose the send buffer sizes
    total_red_size = -1
    key = True
    while key:
        rsizes = torch.zeros(num_workers,dtype=torch.int32,device=tensor.device)
        
        dist.all_to_all_single(rsizes, ssizes)
       
        total_red_size = 0

        key = False
        key1 = torch.BoolTensor([False]).to(device=tensor.device)
        for i in range(num_workers):
            total_red_size+=rsizes[i]
            if rsizes[i] >= 500000000 or total_red_size < 0:
                key1 = torch.BoolTensor([True]).to(device=tensor.device)
                break
            
        key_buffer = [torch.BoolTensor([True]).to(device=tensor.device) for _ in range(num_workers)]
        
        dist.all_gather(key_buffer, key1)

        for i in range(num_workers):
            if key_buffer[i].item():
                key = True
                break

    rsizes = rsizes.cpu().numpy()
    
    local_topk_indexes = np.concatenate(split_topk_indexes)
    if local_topk_indexes.size < 2*topk_value/3: 
        _local_threshold[new_name] /= scale
    elif local_topk_indexes.size > 5*topk_value/4: 
        _local_threshold[new_name] *= scale

    whole_value_rbuffers = np.zeros(total_red_size, dtype='float32')
    whole_index_rbuffers = np.zeros(total_red_size, dtype='int32')

    all_value_rbuffers = []
    all_index_rbuffers = []
    r_roll_rsizes = np.roll(rsizes[::-1], rank+1)

    r_offsets[1:] = r_roll_rsizes[:-1]
    r_offsets = np.cumsum(r_offsets)

    for i in range(num_workers):
        if i < num_workers-1:
            all_index_rbuffers.append(whole_index_rbuffers[r_offsets[i]:r_offsets[i+1]])
            all_value_rbuffers.append(whole_value_rbuffers[r_offsets[i]:r_offsets[i+1]])
        else:
            all_index_rbuffers.append(whole_index_rbuffers[r_offsets[i]: ])
            all_value_rbuffers.append(whole_value_rbuffers[r_offsets[i]: ])

    chunk_offsets = []
    inner_chunk_offsets = []
    inner_chunk_sizes = []
    for i in range(msg_chunks):
        chunk_offsets.append(r_offsets[i*throttle])
        inner_chunk_offsets.append(r_offsets[i*throttle : min((i+1)*throttle, num_workers)] - r_offsets[i*throttle])
        inner_chunk_sizes.append(r_roll_rsizes[i*throttle : min((i+1)*throttle, num_workers)])

    # communicate for the first chunk
    reqs = []
    torch.cuda.set_device(tensor.device)
    for i in range(0, num_workers):
        all_index_sbuffers[i] = torch.tensor(all_index_sbuffers[i], dtype=torch.int32).to(device)
        all_value_sbuffers[i] = torch.tensor(all_value_sbuffers[i], dtype=torch.float32).to(device)
        all_index_rbuffers[i] = torch.tensor(all_index_rbuffers[i], dtype=torch.int32).to(device)
        all_value_rbuffers[i] = torch.tensor(all_value_rbuffers[i], dtype=torch.float32).to(device)

    for i in range(0, throttle):
        dst = dsts[i]
        src = srcs[i]
        if i == 0:
            assert dsts[i] == srcs[i] == rank
            all_value_rbuffers[i][:] = all_value_sbuffers[dsts[i]][:]
            all_index_rbuffers[i][:] = all_index_sbuffers[dsts[i]][:]
        else:
            reqs.append(dist.P2POp(dist.isend,all_index_sbuffers[dst], dst,tag=1))
            reqs.append(dist.P2POp(dist.irecv,all_index_rbuffers[i], src,tag=1))
            reqs.append(dist.P2POp(dist.isend,all_value_sbuffers[dst], dst,tag=2))
            reqs.append(dist.P2POp(dist.irecv,all_value_rbuffers[i], src,tag=2))

    reqs = dist.batch_isend_irecv(reqs)
    for req in reqs:
        req.wait()

    for i in range(0, throttle):
        if i < num_workers-1:
            whole_index_rbuffers[r_offsets[i]:r_offsets[i+1]] = all_index_rbuffers[i].cpu().numpy()
            whole_value_rbuffers[r_offsets[i]:r_offsets[i+1]] = all_value_rbuffers[i].cpu().numpy()
        else:
            whole_index_rbuffers[r_offsets[i]: ] = all_index_rbuffers[i].cpu().numpy()
            whole_value_rbuffers[r_offsets[i]: ] = all_value_rbuffers[i].cpu().numpy()
    
    # communicate for the following chunk with computation overlapping
    for i in range(1, msg_chunks):
        reqs = []
        for j in range(throttle*i, min(num_workers, throttle*(i+1))):
            dst = dsts[j]
            src = srcs[j]
            #exchange buffer
            reqs.append(dist.P2POp(dist.isend,all_index_sbuffers[dst], dst,tag=1))
            reqs.append(dist.P2POp(dist.irecv,all_index_rbuffers[j], src,tag=1))
            reqs.append(dist.P2POp(dist.isend,all_value_sbuffers[dst], dst,tag=2))
            reqs.append(dist.P2POp(dist.irecv,all_value_rbuffers[j], src,tag=2))

        chunk_offset = chunk_offsets[i-1]
        chunk_size = chunk_offsets[i]-chunk_offsets[i-1]
        inner_chunk_offset = inner_chunk_offsets[i-1]
        inner_chunk_size = inner_chunk_sizes[i-1]
        tmp_indexes = torch.from_numpy(whole_index_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(tensor.device, non_blocking=False).long()
        tmp_values = torch.from_numpy(whole_value_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(tensor.device, non_blocking=False)
        for k in range(inner_chunk_offset.size):
            if inner_chunk_size[k] == 0:
                pass
            else:
                reduced_t[tmp_indexes[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]] += tmp_values[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]
        reqs = dist.batch_isend_irecv(reqs)
        for req in reqs:
            req.wait()
        for j in range(throttle*i, min(num_workers, throttle*(i+1))):
            if j < num_workers-1:
                whole_index_rbuffers[r_offsets[j]:r_offsets[j+1]] = all_index_rbuffers[j].cpu().numpy()
                whole_value_rbuffers[r_offsets[j]:r_offsets[j+1]] = all_value_rbuffers[j].cpu().numpy()
            else:
                whole_index_rbuffers[r_offsets[j]: ] = all_index_rbuffers[j].cpu().numpy()
                whole_value_rbuffers[r_offsets[j]: ] = all_value_rbuffers[j].cpu().numpy()
    
    # compute for the last chunk
    chunk_offset = chunk_offsets[msg_chunks-1]
    chunk_size = total_red_size - chunk_offsets[msg_chunks-1]
    inner_chunk_offset = inner_chunk_offsets[msg_chunks-1]
    inner_chunk_size = inner_chunk_sizes[msg_chunks-1]
    tmp_indexes = torch.from_numpy(whole_index_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(tensor.device, non_blocking=False).long()
    tmp_values = torch.from_numpy(whole_value_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(tensor.device, non_blocking=False)
    for k in range(inner_chunk_offset.size):
        if inner_chunk_size[k] == 0:
            pass
        else:
            reduced_t[tmp_indexes[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]] += tmp_values[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]

    reduced = reduced_t.cpu().numpy()
    send_size = torch.zeros(1,dtype=torch.int32, device=tensor.device)
    recv_sizes = [torch.zeros(1,dtype=torch.int32, device=tensor.device) for _ in range(num_workers)]
    offsets = [torch.zeros(1,dtype=torch.int32, device=tensor.device) for _ in range(num_workers)]


    if allreduce_counter[new_name] % global_threshold_recompute_interval == 0:
        gindexes = np.nonzero(reduced)[0]
        gvalues = reduced[gindexes]
        send_size[0] = gvalues.size * 2
        dist.all_gather(recv_sizes, send_size)
        offsets[1:] = recv_sizes[:-1]
        offsets = torch.cat(offsets).to(device=tensor.device)
        offsets = torch.cumsum(offsets, dim=0)
        total_size = 0
        for i in range(num_workers):
            total_size += recv_sizes[i].item()
        recv_buffer = torch.zeros(total_size, dtype=torch.float32, device=tensor.device)

        send_buffer = np.zeros(send_size[0], dtype='float32')
        send_buffer[0 : send_size[0]//2] = gindexes.astype(np.int32)
        send_buffer[send_size[0]//2 : send_size[0]] = gvalues.astype(np.float32)
        send_tensor = torch.from_numpy(send_buffer).to(device=tensor.device)
        length_buffer = [None for _ in range(group_to_use.size())]
        dist.all_gather_object(length_buffer, send_tensor.numel(), group=group_to_use)
        max_length = max(length_buffer)
        if send_tensor.numel() < max_length:
            send_tensor = torch.cat([send_tensor.to(tensor.device),torch.zeros([max_length - send_tensor.numel()], device=tensor.device)])

        recv_tensors = [torch.zeros_like(send_tensor, device=tensor.device) for _ in range(group_to_use.size())]
        dist.all_gather(recv_tensors, send_tensor.to(tensor.device), group=group_to_use, async_op=False)

        for i in range(num_workers):
            recv_buffer[offsets[i]:offsets[i]+recv_sizes[i]] = recv_tensors[i][:recv_sizes[i]]

        recv_buffer = recv_buffer.cpu().numpy()

        all_gindexes = np.zeros(total_size//2, dtype='int32')
        all_gvalues = np.zeros(total_size//2, dtype='float32')
        for i in range(num_workers):
            offset = offsets[i]//2
            size = recv_sizes[i]//2
            all_gindexes[offset:offset+size] = recv_buffer[offsets[i]:offsets[i]+size].astype(np.int32) + region_offsets[i]
            all_gvalues[offset:offset+size] = recv_buffer[offsets[i]+size:offsets[i]+2*size].astype(np.float32)

        with torch.no_grad():
            all_gindexes_tensor = torch.from_numpy(all_gindexes).to(device=tensor.device).long()
            all_gvalues_tensor = torch.from_numpy(all_gvalues).to(device=tensor.device)
            gtopk_values, gtopk_values_indexes, global_threshold = k2globalthreshold(all_gvalues_tensor, max(topk_value, 1))
            gtopk_gindexes_tensor = all_gindexes_tensor[gtopk_values_indexes]
            gtopk_values /= num_workers 
            result = tensor
            result.data.fill_(0.)
            result[gtopk_gindexes_tensor] = gtopk_values
            gtopk_gindexes_tensor = gtopk_gindexes_tensor.type(torch.IntTensor)
    
        gtopk_gindexes = gtopk_gindexes_tensor.cpu().numpy()
        involved_indexes = np.intersect1d(local_topk_indexes, gtopk_gindexes, return_indices=False, assume_unique=True)
        indexes_t = torch.from_numpy(involved_indexes).to(device=layers[new_name].device).long()
        layers[new_name][indexes_t] = 0.0
        _global_threshold[new_name] = global_threshold
    else:
        with torch.no_grad():
            reduced_tensor = torch.from_numpy(reduced).to(device=tensor.device)
            gindexes, gvalues = compressbythreshold(tensor=reduced_tensor, thres=_global_threshold[new_name])        
            gindexes = gindexes.cpu().numpy()
        gvalues = gvalues.cpu().numpy()
        send_size[0] = gvalues.size * 2
        dist.all_gather(recv_sizes,send_size)

        offsets[1:] = recv_sizes[:-1]
        offsets = torch.cat(offsets).to(device=tensor.device)
        offsets = torch.cumsum(offsets,dim = 0)
        total_size = 0
        for i in range(num_workers):
            total_size+=recv_sizes[i].item()

        send_buffer = np.zeros(send_size[0], dtype='float32')
        send_buffer[0 : send_size[0]//2] = gindexes.astype(np.int32)
        send_buffer[send_size[0]//2 : send_size[0]] = gvalues.astype(np.float32)
        
        allgather_recv_buffer = torch.zeros(total_size, dtype=torch.float32, device=tensor.device)
        send_tensor = torch.from_numpy(send_buffer).to(device=tensor.device)

        length_buffer = [None for _ in range(group_to_use.size())]


        dist.all_gather_object(length_buffer, send_tensor.numel(), group=group_to_use)
        max_length = max(length_buffer)
        if send_tensor.numel() < max_length:
            send_tensor = torch.cat([send_tensor.to(tensor.device),torch.zeros([max_length - send_tensor.numel()], device=tensor.device)])

        recv_tensors = [torch.zeros_like(send_tensor, device=tensor.device) for _ in range(group_to_use.size())]
        dist.all_gather(recv_tensors, send_tensor.to(tensor.device), group=group_to_use, async_op=False)

        for i in range(num_workers):
            allgather_recv_buffer[offsets[i]:offsets[i]+recv_sizes[i].item()] = recv_tensors[i][:recv_sizes[i].item()]

        allgather_recv_buffer = allgather_recv_buffer.cpu().numpy()
        offsets = offsets.cpu().numpy()
        recv_sizes = torch.cat(recv_sizes).to(device=tensor.device)
        recv_sizes = recv_sizes.cpu().numpy()

        all_gindexes = np.zeros(total_size//2, dtype='int32')
        all_gvalues = np.zeros(total_size//2, dtype='float32')
        for i in range(num_workers):
            offset = offsets[i]//2
            size = recv_sizes[i]//2
            all_gindexes[offset:offset+size] = allgather_recv_buffer[offsets[i]:offsets[i]+size].astype(np.int32) + region_offsets[i]
            all_gvalues[offset:offset+size] = allgather_recv_buffer[offsets[i]+size:offsets[i]+2*size].astype(np.float32)

        involved_indexes = np.intersect1d(local_topk_indexes, all_gindexes, return_indices=False, assume_unique=True)
        indexes_t = torch.from_numpy(involved_indexes).to(device=layers[new_name].device).long()
        layers[new_name][indexes_t] = 0.0

        if all_gindexes.size < 2*topk_value/3: 
            _global_threshold[new_name] /= 1.008
        elif all_gindexes.size > 4*topk_value/3: 
            _global_threshold[new_name] *= 1.008

        with torch.no_grad():
            all_gindexes_tensor = torch.from_numpy(all_gindexes).to(device=tensor.device).long()
            all_gvalues_tensor = torch.from_numpy(all_gvalues).to(device=tensor.device)

            all_gvalues_tensor /= num_workers 
            result = tensor
            result.data.fill_(0.)
            result[all_gindexes_tensor] = all_gvalues_tensor

    allreduce_counter[new_name] += 1

    fut = torch.futures.Future()

    fut.set_result([result])

    return fut


def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensors to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.get_tensor().to(torch.float16).div_(world_size)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.get_tensor()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return [decompressed_tensor]

    return fut.then(decompress)


def fp16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future]
) -> Callable[[Any, dist.GradBucket], torch.futures.Future]:
    """
    This wrapper casts the input gradient tensors of a given DDP communication hook to half-precision
    floating point format (``torch.float16``), and casts the resulting tensors of the given hook back to
    the input data type, such as ``float32``.

    Therefore, ``fp16_compress_hook`` is equivalent to ``fp16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, fp16_compress_wrapper(powerSGD_hook))
    """

    def fp16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future:
        # Cast bucket tensor to the FP16.
        bucket.set_tensor(bucket.get_tensor().to(torch.float16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.get_tensor()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value()[0])
            return [decompressed_tensor]

        # Decompress after hook has run.
        return fut.then(decompress)

    return fp16_compress_wrapper_hook

def _allgather_topk(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    k = int(len(tensor) * 0.01)
    result = torch.topk(tensor.abs(), k)
    indices_buffer = [torch.zeros_like(result.indices, device=tensor.device) for _ in range(group_to_use.size())]
    values_buffer = [torch.zeros_like(result.values, device=tensor.device) for _ in range(group_to_use.size())]
    
    fut1 = dist.all_gather(indices_buffer, result.indices.to(tensor.device), group=group_to_use, async_op=True).get_future()
    fut = dist.all_gather(values_buffer, tensor[result.indices], group=group_to_use, async_op=True).get_future()

    global layers,key_ef,bucket_number
    def store_small_gradients():
        global num_step
        tensor[indices_buffer[torch.distributed.get_rank()]] = 0
        #layers[str(bucindex)] = tensor*(0.5+0.1*int(num_step/(bucket_number*100))) # EF scheduler
        layers[str(bucindex)] = tensor
        if num_step < bucket_number*500:
            num_step+=1

    def div_by_group_size(fut):
        global layers,key_ef,bucket_number
        if key_ef >= bucket_number:
            store_small_gradients()
        else:
            key_ef += 1
        tensor.zero_()
        for i in range(group_to_use.size()):
            tensor.scatter_add_(0, indices_buffer[i], values_buffer[i])
        return [tensor.div_(group_to_use.size())]

    return fut.then(div_by_group_size)

def _allgather_randomk(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    k = int(len(tensor) * 0.01)
    torch.manual_seed(num_step)
    indices = torch.randperm(tensor.numel(), device=tensor.device)[:k]
    values = tensor[indices]
    indices_buffer = [torch.zeros_like(indices, device=tensor.device) for _ in range(group_to_use.size())]
    values_buffer = [torch.zeros_like(values, device=tensor.device) for _ in range(group_to_use.size())]
    fut = dist.all_gather(values_buffer, values, group=group_to_use, async_op=True).get_future()
    fut1 = dist.all_gather(indices_buffer, indices.to(tensor.device), group=group_to_use, async_op=True).get_future()

    global layers,key_ef,bucket_number
    def store_small_gradients():
        global num_step
        tensor[indices_buffer[torch.distributed.get_rank()]] = 0
        layers[str(bucindex)] = tensor
        num_step+=1

    def div_by_group_size(fut):
        global layers,key_ef,bucket_number
        if key_ef >= bucket_number:
            store_small_gradients()
        else:
            key_ef += 1
        tensor.zero_()
        for i in range(group_to_use.size()):
            tensor.scatter_add_(0, indices_buffer[i], values_buffer[i])
        return [tensor.div_(group_to_use.size())]

    return fut.then(div_by_group_size)

def _allgather_efsignsgd(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    shape = tensor.size()
    tensor = tensor.flatten()

    sign_encode = tensor >= 0
    mean = tensor.abs().mean()
    #tensor_compressed = mean, sign_encode.type(torch.uint8)
    mean_buffer = [torch.zeros_like(mean, device=tensor.device) for _ in range(group_to_use.size())]
    sign_encode_buffer = [torch.zeros_like(sign_encode.type(torch.uint8), device=tensor.device) for _ in range(group_to_use.size())]
    fut1 = dist.all_gather(mean_buffer, mean.to(tensor.device), group=group_to_use, async_op=True).get_future()
    fut = dist.all_gather(sign_encode_buffer, sign_encode.type(torch.uint8).to(tensor.device), group=group_to_use, async_op=True).get_future()
    
    def div_by_group_size(fut):
        global layers,key_ef,bucket_number
        if key_ef >= bucket_number:
            sign_decode = sign_encode_buffer[torch.distributed.get_rank()].type(torch.float32) * 2 - 1
            sign_decode = mean_buffer[torch.distributed.get_rank()].item() * sign_decode
            tensor_decompressed = sign_decode.view(tensor.size())
            layers[str(bucindex)] = tensor - tensor_decompressed
        else:
            key_ef += 1

        temp = tensor.zero_()
        for i in range(group_to_use.size()):
            sign_decode = sign_encode_buffer[i].type(torch.float32) * 2 - 1
            sign_decode = mean_buffer[i].item() * sign_decode
            tensor_decompressed = sign_decode.view(tensor.size())
            temp+=tensor_decompressed
        return [temp.div_(group_to_use.size())]

    return fut.then(div_by_group_size)

def _allgather_dgc(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    shape = tensor.size()
    tensor = tensor.flatten()
    numel = tensor.numel()

    sample_shape = [max(1, int(numel * 0.01))]
    sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
    sample_tensor = tensor[sample_index]
    compression = 0.01

    k = max(1, int(numel * compression * 0.01))
    vals, indices = torch.topk(sample_tensor.abs(), k)

    threshold = vals.min()
    mask = tensor.abs() >= threshold
    selected = mask.sum()

    for _ in range(10):
        if selected > 1.3 * numel * compression:
            threshold = 1.3 * threshold
            mask = tensor.abs() >= threshold
            selected = mask.sum()
        elif selected < 0.7 * numel * compression:
            threshold = 0.7 * threshold
            mask = tensor.abs() >= threshold
            selected = mask.sum()
        else:
            break

    values = tensor[mask]
    length_buffer = [None for _ in range(group_to_use.size())]
    dist.all_gather_object(length_buffer, values.numel(), group=group_to_use)
    max_length = max(length_buffer)
    if values.numel() < max_length:
        values = torch.cat([values.to(tensor.device),torch.zeros([max_length - values.numel()], device=tensor.device)])
        indices = torch.cat([mask.to(tensor.device),torch.ones([max_length - values.numel()], device=tensor.device)])

    indices_buffer = [torch.zeros_like(indices, device=tensor.device) for _ in range(group_to_use.size())]
    values_buffer = [torch.zeros_like(values, device=tensor.device) for _ in range(group_to_use.size())]
    dist.all_gather(indices_buffer, indices.to(tensor.device), group=group_to_use, async_op=False)
    fut = dist.all_gather(values_buffer, values.to(tensor.device), group=group_to_use, async_op=True).get_future()

    global layers,key_ef,bucket_number
    def store_small_gradients():
        global layers
        tensor[indices_buffer[torch.distributed.get_rank()].long()] = 0
        layers[str(bucindex)] = tensor

    def div_by_group_size(fut):
        global layers,key_ef,bucket_number
        if key_ef >= bucket_number:
            store_small_gradients()
        else:
            key_ef += 1
        tensor.zero_()
        for i in range(group_to_use.size()):
            tensor.scatter_add_(0, indices_buffer[i].long(), values_buffer[i])
        return [tensor.div_(group_to_use.size())]

    return fut.then(div_by_group_size)
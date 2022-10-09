import torch
import torch.distributed as dist
import os
import torch.nn as nn
import time
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
from tqdm import tqdm, trange
import math

from scheduler import PolynomialLRDecay

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

local_rank = args.local_rank

torch.cuda.set_device(local_rank)

dist.init_process_group(backend='nccl', init_method='env://')
print('ok')

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomResizedCrop(224),		#对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),		
    transforms.ToTensor(),					
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))	
])

transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# model_name='resnet50'
# resnet_50 = torchvision.models.resnet50()
# resnet_50.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
# model = resnet_50.to(local_rank)

model_name='resnet101'
resnet_101 = torchvision.models.resnet101(pretrained = True)
#resnet_101.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
model = resnet_101.to(local_rank)
ex='test'
batch_size = 32
epochs = 10

# model_name='vgg19'
# vgg_19 = torchvision.models.vgg19(pretrained = False)
# vgg_19.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))
# model = vgg_19.to(local_rank)
# ex='test'
# batch_size = 64
# epochs = 30

dataset = torchvision.datasets.CIFAR10
trainset = dataset(root='./data', train=True, download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            sampler=train_sampler,
                                            shuffle=False)

testset = dataset(root='./data', train=False, download=True,
                      transform=transform_test)
test_sampler = torch.utils.data.SequentialSampler(testset)                      
test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=64,
                                              sampler=test_sampler,
                                              shuffle=False)



def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)

ddp_model = DDP(model, device_ids=[local_rank],bucket_cap_mb=25) #for VGG-19, use bucket_cap_mb=15
import ddp_comm_hooks_new
ddp_model.register_comm_hook(state=None, hook=ddp_comm_hooks_new.default_hooks.allreduce_hook)
# ddp_model.register_comm_hook(dist.group.WORLD, ddp_comm_hooks_new.default_hooks.fp16_compress_hook)
# state = ddp_comm_hooks_new.powerSGD_hook.PowerSGDState(process_group=dist.group.WORLD, matrix_approximation_rank=1,
#                                   start_powerSGD_iter=10, min_compression_rate=0.5)
# ddp_model.register_comm_hook(state, ddp_comm_hooks_new.powerSGD_hook.powerSGD_hook)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(ddp_model.parameters(), lr=0.001,momentum=0.9, weight_decay=1e-4)

poly_decay_scheduler = PolynomialLRDecay(optimizer=optimizer, max_decay_steps=epochs * len(train_loader),end_learning_rate=0.0001, power=2.0)
start_epoch = 0
start_time = 0

import time

facc = open(str(batch_size)+'_cifar_'+model_name+'_'+ex+'_accuracy.txt','w')
floss = open(str(batch_size)+'_cifar_'+model_name+'_'+ex+'_loss.txt','w')
ftime = open(str(batch_size)+'_cifar_'+model_name+'_'+ex+'_epochtime.txt','w')
testtime = 0
def test():
    global testtime
    t = time.time()
    #ddp_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    iter_bar = tqdm(test_loader, desc="iter", disable=False)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(iter_bar):
            ddp_model.eval()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = ddp_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        print('acc:',acc)
    testtime += time.time() - t
    facc.write(str(time.time()-testtime-starttime)+'\t'+str(acc)+'\n' )
    facc.flush()
    return

t = time.time()
starttime = time.time()
st = 0
#print(start_epoch,start_time)
for _ in trange(start_epoch, int(epochs), desc="Epoch", disable=False):
    iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
    epoch_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(iter_bar):
        inputs, targets = inputs.to(local_rank), targets.to(local_rank)
        optimizer.zero_grad()
        # if st == 1:
        #     with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA,
        #         ]
        #     ) as p1:
        #         outputs = ddp_model(inputs)
        #     p1.export_chrome_trace(f'trace/forward-rank'+str(dist.get_rank())+'.trace')
        # else:
        
        # if st == 3:
        #     with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA,
        #         ]
        #     ) as p:
        #         outputs = ddp_model(inputs)
        #         loss = loss_fn(outputs, targets)
        #         loss.backward()
        #     p.export_chrome_trace(f'trace/base-rank'+str(dist.get_rank())+'.trace')
        # else:
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        floss.write(str(loss.item() )+'\n' )
        iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())

        poly_decay_scheduler.step()
        st += 1
    if dist.get_rank() == 0:
        test()
    ftime.write(str(time.time()-epoch_time)+'\n')
    
    print(time.time()- t - testtime + start_time)
    floss.flush()
    ftime.flush()
facc.close()
floss.close()
ftime.close()

print(time.time()- t - testtime)
torch.save(model, model_name+'-'+ex+'.pth')

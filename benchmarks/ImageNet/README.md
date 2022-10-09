# ImageNet training in PyTorch
referred to https://github.com/pytorch/examples/tree/main/imagenet

## use OGC
re-compile pytorch (see "torch" folder) and use _allreduce_fut hook in ddp_comm_hooks_new to trigger it.

## baseline
use original pytorch and _allreduce_fut hook.

## script
imagenet-test-meng.sh
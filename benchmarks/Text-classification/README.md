# Bert-Chinese-Text-Classification-Pytorch
referred to https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

## use OGC
re-compile pytorch (see "torch" folder) and use _allreduce_fut hook in ddp_comm_hooks_new to trigger it.

## other GC schemes
use original pytorch and use the hook you want (see notation in default_hooks.py and run.py). baseline is also using _allreduce_fut hook.

## script
bert-test-meng.sh
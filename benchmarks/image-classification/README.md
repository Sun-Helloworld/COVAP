# Image-Classification-Pytorch

## use OGC
re-compile pytorch (see "torch" folder) and use _allreduce_fut hook in ddp_comm_hooks_new to trigger it.

## other GC schemes
use original pytorch and use the hook you want (see notation in default_hooks.py and ddp-test-meng.py). baseline is also using _allreduce_fut hook.

## script
cv-test-meng.sh
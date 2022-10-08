## Use OGC
Replace the correspond files of the source code of Pytorch by those files we provided. Note that in other version those files may be in different directories.

We also provide our script for easy compiling on multi-node environment. (You need to run compile_pytorch.sh on all your machines.) 

The core code of OGC is in "torch/lib/c10d/reducer.cpp".

If you do not want to recompile Pytorch, you can also easily re-implemented OGC in Pytorch DDP communication hooks. However, the speedups may be much lower than implemented in DDP module.

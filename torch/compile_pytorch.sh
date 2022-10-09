source /root/anaconda3/bin/activate  /root/anaconda3/envs/Your_name

scp root@$MasterIPAddress:$SourceCodePath/pytorch/torch/lib/c10d/reducer.cpp $SourceCodePath/pytorch/torch/lib/c10d/
scp root@$MasterIPAddress:$SourceCodePath/pytorch/torch/lib/c10d/reducer.hpp $SourceCodePath/pytorch/torch/lib/c10d/
scp root@$MasterIPAddress:$SourceCodePath/pytorch/torch/lib/c10d/comm.hpp $SourceCodePath/pytorch/torch/lib/c10d/
scp root@$MasterIPAddress:$SourceCodePath/pytorch/torch/csrc/distributed/c10d/init.cpp $SourceCodePath/pytorch/torch/csrc/distributed/c10d

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cd $SourceCodePath/pytorch

python setup.py install

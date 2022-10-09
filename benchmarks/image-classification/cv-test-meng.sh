scp root@$MasterIPAddress:$CodePath/ddp-test-meng.py $CodePath
scp root@$MasterIPAddress:$CodePath/ddp_comm_hooks_new/default_hooks.py $CodePath/ddp_comm_hooks_new


if ! [[ $(nvidia-smi) =~ "No running processes found" ]]; then
  echo "There are other processes using GPU, so this process exits. The related processes are as follows:"
  nvi=$(nvidia-smi)
  res=$(grep -n '+-----------------------------------------------------------------------------+' <<< "$nvi" | awk -F: '{print $1}' | tail -n2)
  lineBegin=$(head -n1 <<< "$res")
  lineEnd=$(tail -n1 <<< "$res")
  processes=$(head -n $((lineEnd-1))<<< "$nvi" | tail -n $((lineEnd-lineBegin-5)))
  ps f $(awk '{print $5}' <<<"$processes")
  exit 1    
fi

source /root/anaconda3/bin/activate  /root/anaconda3/envs/Your_name

declare master_pids
declare rank=$NODE_RANK

echo "(" cd $CodePath "&&" NCCL_SOCKET_IFNAME="eth0" GLOO_SOCKET_IFNAME="eth0" python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=${rank} --master_addr=$MasterIPAddress --master_port=23579 ddp-test-meng.py ")"
( cd $CodePath && NCCL_SOCKET_IFNAME="eth0" GLOO_SOCKET_IFNAME="eth0" python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=${rank} --master_addr=$MasterIPAddress --master_port=23579 ddp-test-meng.py  ) &
master_pids=$!

# Ctrl+C
handleSigInt() {
  echo "正在杀死所有进程"
  ps aux|grep -E "ddp-test-meng.py"|grep -v grep|awk '{print $2}'|xargs kill -9
  exit 0 
}

trap handleSigInt SIGINT

wait ${master_pids}

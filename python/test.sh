
source python-env/bin/activate
source ../../../openvino/build/install/setupvars.sh # Your OV Env.
# source /mnt/disk2/xiping_tmp/openvino/build/install/setupvars.sh

# OV_CPU_DEBUG_LOG=- 
numactl -C 0-47 python model_gather_embedding.py
# gdb --args
#  > cpu_debug.log

source ../../python-env/bin/activate
source ../../openvino/build/install/setupvars.sh # Your OV Env.
# source ../../openvino/build_debug/install/setupvars.sh # Your OV Env.
# source /mnt/disk1/xiping/openvino/build/install/setupvars.sh

# master
# source /mnt/data_nvme1n1p1/xiping_workpath/openvino/build/install/setupvars.sh

# Maxsim OV: remove assign.
# source /mnt/data_nvme1n1p1/xiping_workpath/ov2/openvino/build/install/setupvars.sh

echo "================================="
echo "Tip:"
echo "ENV: GDB=1, just for gdb --args"
echo "ENV: RUN1=1, just runn one time inference."
echo "ENV: EXECUTE_LOG=1, dump execution log via: export OV_CPU_DEBUG_LOG=-"
echo "================================="

if [[ -z ${GDB} ]]; then
    echo "-->Not debug with gdb."
else
    echo "-->Debug with gdb."
    GDB="gdb --args "
fi

if [[ -z ${EXECUTE_LOG} ]]; then
    echo "-->Not dump execution log."
else
    echo "-->Dump execution log: OV_CPU_DEBUG_LOG=- "
    export OV_CPU_DEBUG_LOG=-
fi

# OV_CPU_DEBUG_LOG=- 

# numactl -C 0-15 $GDB python ./model_conv_bias_sum_reshape.py

# numactl -C 96-137 python model_gather_embedding.py
# numactl -C 0-47 $GDB python model_gather_embedding.py
# numactl -C 0-47 $GDB python3 ./compare_result_and_expected.py
# numactl -C 0-47 $GDB python model_gather_embedding_versa.py

# numactl -C 0-47 $GDB python model_if.py
# export OV_CPU_EXEC_GRAPH_PATH=xxx.xml 
# export OV_CPU_DEBUG_LOG=-
# export ENABLE_RVSUBGRAPH=1

numactl -C 0-15 $GDB python model_stateful_readvalue_assign.py
#numactl -C 0-15 $GDB python model_readvalue_init_subgraph.py

# OV_CPU_EXEC_GRAPH_PATH=xxx.xml numactl -C 0-15 $GDB python model_gather.py

# numactl -C 0-15 $GDB python model_add.py



source python-env/bin/activate
# source ../../../openvino/build/install/setupvars.sh # Your OV Env.
# source /mnt/disk2/xiping_tmp/openvino/build/install/setupvars.sh
source /mnt/data_nvme1n1p1/xiping_workpath/golubev_ov/openvino/build/install/setupvars.sh

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
    DEBUG_GDB="gdb --args "
fi

if [[ -z ${EXECUTE_LOG} ]]; then
    echo "-->Not dump execution log."
else
    echo "-->Dump execution log: OV_CPU_DEBUG_LOG=- "
    export OV_CPU_DEBUG_LOG=-
fi

# OV_CPU_DEBUG_LOG=- 
# numactl -C 96-137 python model_gather_embedding.py
# numactl -C 0-47 $DEBUG_GDB python model_gather_embedding.py
numactl -C 0-47 $DEBUG_GDB python3 ./compare_result_and_expected.py

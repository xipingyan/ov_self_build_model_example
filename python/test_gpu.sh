
source ../python-env/bin/activate
source ../../openvino/build/install/setupvars.sh # Your OV Env.

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

export OV_DEVICE='GPU'

# Test: model_matmul.py
numactl -C 0-15 $GDB python model_matmul.py

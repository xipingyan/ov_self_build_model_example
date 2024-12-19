
# source /opt/intel/oneapi/setvars.sh
source ../python-env/bin/activate
source ../../openvino/build/install/setupvars.sh # dpc++
# source ../../ov_orig/openvino/build/install/setupvars.sh # dpc++
# source /mnt/xiping/openvino/build/install/setupvars.sh


echo "================================="
echo "Tip:"
echo "ENV: GDB=1, just for gdb --args"
echo "================================="

if [[ -z ${GDB} ]]; then
    echo "-->Not debug with gdb."
else
    echo "-->Debug with gdb."
    GDB="gdb --args "
fi

export OV_DEVICE='GPU'

# Remove old cache npy data.
# rm -rf input.npy weight.npy

# Some debug macro
# export OV_GPU_Help=1
# export OV_GPU_DisableOnednn=1     # Mutmal result is wrong. I don't know why?
# export OV_GPU_ForceImplTypes=matmal:ocl
# export OV_GPU_ForceImplTypes=matmal:onednn

# Debug log:
# DISABLED = 0,
# INFO = 1,
# LOG = 2,
# TRACE = 3,
# TRACE_DETAIL = 4
# OV_GPU_Verbose=4

# Test: model_matmul.py
OV_GPU_Verbose=4 numactl -C 0-15 $GDB python model_matmul.py

# source ../../openvino/build_gcc/install/setupvars.sh # Your OV Env.
# numactl -C 0-15 $GDB python model_matmul.py

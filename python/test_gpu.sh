
source ../python-env/bin/activate
source ../../openvino/build/install/setupvars.sh # dpc++

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
export OV_GPU_DisableOnednn=1
# export OV_GPU_ForceImplTypes=matmal:ocl

# Test: model_matmul.py
numactl -C 0-15 $GDB python model_matmul.py

# source ../../openvino/build_gcc/install/setupvars.sh # Your OV Env.
# numactl -C 0-15 $GDB python model_matmul.py

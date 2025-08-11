source ../../python-env/bin/activate
# source /mnt/xiping/mygithub/openvino/build/install/setupvars.sh
source /mnt/xiping/openvino/build/install/setupvars.sh

# onetrace --chrome-call-logging --chrome-device-timeline
python test_register_custom_op.py

# ROOTPATH=/mnt/xiping/mygithub/ov_self_build_model_example/python/custom_op/1_register_kernel
# # model=./export_ov_model/openvino_model_GPU_static.xml
# model=./export_ov_model/openvino_model_GPU_dynamic.xml
# /mnt/xiping/mygithub/openvino/bin/intel64/Debug/benchmark_app -m $model -d GPU -nstreams 1 -nthreads 1 -hint none -niter 1 \
#     -c ${ROOTPATH}/gpu/custom_add.xml \
#     -extensions ${ROOTPATH}/cpu/build/libopenvino_custom_add_extension.so \
#     -data_shape input[1,5,10]
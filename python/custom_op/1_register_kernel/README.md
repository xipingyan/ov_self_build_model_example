# Register costom kernel

Refer: https://docs.openvino.ai/2025/documentation/openvino-extensibility/custom-gpu-operations.html#

####  How to run.

1. Build cpp custom kernel(MyAdd)

    cd cpu/ && make build && cd build
    source openvino/build/install/setupvars.sh  # Source your self built openvino.
    cmake ..
    make -j20
    
    ls *.so
    libopenvino_custom_add_extension.so

2. Run pipeline

    source ../../python-env/bin/activate
    python test_register_custom_op.py
    
    <!-- log -->
    == Start to run OV model.
    == Compare result, Torch VS OV = True

3. Benchmarkapp test custom kernel IR

    ROOTPATH=/mnt/xiping/mygithub/ov_self_build_model_example/python/custom_op/1_register_kernel
    model=./export_ov_model/openvino_model_GPU_static.xml
    app=/mnt/xiping/mygithub/openvino/bin/intel64/Debug/benchmark_app

    $app -m $model -d GPU -nstreams 1 -nthreads 1 -hint none -niter 1 \
        -c ${ROOTPATH}/gpu/custom_add.xml \
        -extensions ${ROOTPATH}/cpu/build/libopenvino_custom_add_extension.so
    
    -c: gpu custom op kernel.
    -extensions: cpu kernel implementation.

#### 2 Custom OP

```
./test_2_custom_op.sh
```

#### Custom OP with 2 output

```
./run_custom_op_2_outputs.sh
```


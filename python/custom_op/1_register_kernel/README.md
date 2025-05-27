# Register costom kernel

Refer: https://docs.openvino.ai/2025/documentation/openvino-extensibility/custom-gpu-operations.html#

####  How to run.

1. Build cpp custom kernel(MyAdd)

    cd cpu/ && make build && cd build
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
# README
Just demo how to enable custom op inside OpenVINO pipeline.

There 2 ways to eanble custom op.
1. Register custom kernel. Follow guide: https://docs.openvino.ai/2025/documentation/openvino-extensibility/custom-gpu-operations.html#, 
2. Mapping custom kernel to mulitple OV OPs. Follow guide: https://blog.openvino.ai/blog-posts/openvino-frontend-extension-samples. 

Note: For method 1, there is limitation: no support dynamic shape kernel, because OV opencl kernel shape is decided in compile model.

# 1: Register costom kernel

    Official example: https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations
    

# 2: Mapping custom kernel to mulitple OV OPs

# 3: ...
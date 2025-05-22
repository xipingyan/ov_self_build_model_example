# README
Just demo how to enable custom op inside OpenVINO pipeline.

There 2 ways to eanble custom op.
1. Follow guide: https://docs.openvino.ai/2025/documentation/openvino-extensibility/custom-gpu-operations.html#, register custom kernel;
2. Follow guide: https://blog.openvino.ai/blog-posts/openvino-frontend-extension-samples. Mapping custom kernel to mulitple OV OPs;

Note: For method 1, there is limitation: no support dynamic shape kernel, because OV opencl kernel shape is decided in compile model.

# 1: Register costom kernel

# 2: Convert to multiple OV OPs

# 3: ...
import openvino.runtime.opset11 as ops
from openvino.runtime import Core, Model, PartialShape, Shape, Tensor
import numpy as np

# 测试一个subgraph，功能如下：
# 已知一个tensor[-1, -1], 和一个常量c，生成一个新的tensor[-1, 1]，生成的tensor的第二维度上的值为常量c
def func_broadcast_constant_c(input_tensor, c_value):
    # 1. 获取输入 Tensor 的实际形状 (ShapeOf)
    input_shape = ops.shape_of(input_tensor)

    # 2. 提取第一维 (Batch Size)->shape[1]
    batch_dim = ops.gather(
        data=input_shape, 
        indices=ops.constant([0], dtype=np.int64), 
        axis=ops.constant([0], dtype=np.int64)
    )
    print("batch_dim=", batch_dim)

    # 3. 构建目标形状：[batch_size, 1]
    const_one = ops.constant([1], dtype=np.int64)
    # Shape = [2], value=[batch_size, 1]
    target_shape = ops.concat([batch_dim, const_one], axis=0)
    print("target_shape=", target_shape)

    # 4. 准备常量 c 并在目标形状上进行广播
    c_node = ops.constant(c_value, dtype=np.float32)
    result = ops.broadcast(c_node, target_shape) # Shape = [-1, 1], value=[[c_value], [c_value], ...]
    print("result=", result)
    return result

def create_broadcast_model(c_value):
    # 1. 定义输入 Tensor，形状为 [-1, -1] (动态维度)
    data = ops.parameter(PartialShape([-1, -1]), name="input_tensor", dtype=np.float32)

    result = func_broadcast_constant_c(data, c_value)

    # 6. 封装成 OpenVINO Model
    model = Model([result], [data], "broadcast_c_model")
    return model

# --- 测试运行 ---
core = Core()
c_constant = 99.0
ov_model = create_broadcast_model(c_constant)
compiled_model = core.compile_model(ov_model, "CPU")

# 模拟一个 3x5 的输入 Tensor
test_input = np.ones((3, 5), dtype=np.float32)
res = compiled_model([test_input])[0]

print(f"输入形状: {test_input.shape}")
print(f"输出形状: {res.shape}")
print(f"输出内容:\n{res}")

# 预期输出:
# 输入形状: (3, 5)
# 输出形状: (3, 1)
# 输出内容: [[99.], [99.], [99.]]
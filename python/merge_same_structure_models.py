# 验证如何把2个结构相同的模型串联成一个模型（即 model_1 的输出接到 model_2 的输入上）。注意：Model.replace_parameter 只能替换 Parameter（按index），不能直接用 Result/Output。
# 这里通过“替换 parameter 的 consumers”把 model_2 的输入接到 model_1 的输出上。

# ===== model1 =====
# input1, input2 
#   |       |
# conv1    /
#    \    /
#     add1
#      |
#    output1

# ===== model2 =====
# input1, input2
#   |       |
# conv2    /
#    \    /
#     add2
#      |
#    output2

# ===== merged model =====
# input1, input2
#   |       |
# conv1    / \
#    \    /  |
#     add1   |
#      |     |
#     conv2  |
#       \    /
#        add2
#         |
#       output2

from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import os

def const(shape):
    np.random.seed(42)
    w = np.random.rand(*shape).astype(np.float32)
    return op.Constant(w)

def _model(weight, bias):
    input1 = opset.parameter([-1, 6, -1, -1], Type.f32, name = 'in1')
    input2 = opset.parameter([1, 6], Type.f32, name = 'in2')
   
    b2 = opset.reshape(input2, op.Constant(Type.i32, Shape([4]), [1,6,1,1]), special_zero=False)
 
    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    dilations = [1, 1]
    
    conv = opset.convolution(input1, weight, strides, pads_begin, pads_end, dilations)
    conv = opset.add(conv, b2)
    Result = opset.result(conv, name='output')
    return Model([Result], [input1,input2], 'Model_test')

def _replace_node_consumers(node, new_source_output):
        # Prefer Output.replace if available in this OV build.
        try:
            # =========================================================================
            node.output(0).replace(new_source_output)
            # =========================================================================
            print("Current case will not fallback to replace_source_output.")
            return
        except Exception:
            pass

        # Fallbacks for other OV Python API layouts.
        try:
            from openvino.runtime import replace_source_output  # type: ignore
            replace_source_output(node.output(0), new_source_output)
            return
        except Exception:
            pass

        try:
            from openvino.runtime.op import util as _op_util  # type: ignore
            if hasattr(_op_util, "replace_source_output"):
                _op_util.replace_source_output(node.output(0), new_source_output)
                return
        except Exception:
            pass

        raise TypeError(
            "Cannot rewire model inputs: no supported replace API found. "
            "Try a newer OpenVINO build exposing Output.replace/replace_source_output."
        )

def _merge_model(model_1:Model, model_2:Model):
    # 串联：merged(inputs) -> model_1 -> model_2 -> merged(output)
    # 注意：Model.replace_parameter 只能替换 Parameter（按index），不能直接用 Result/Output。
    # 这里通过“替换 parameter 的 consumers”把 model_2 的输入接到 model_1 的输出上。

    # IMPORTANT: model_1.output(0) is a Result op output; do not feed Result into another op.
    # Use the value feeding the Result instead.
    model1_out = model_1.get_results()[0].input_value(0)
    model1_p0, model1_p1 = model_1.get_parameters()
    model2_p0, model2_p1 = model_2.get_parameters()

    # model_2.input0 <- model_1.output0
    _replace_node_consumers(model2_p0, model1_out)
    # model_2.input1 <- model_1.input1 (second parameter)
    _replace_node_consumers(model2_p1, model1_p1.output(0))

    merged_model = Model([model_2.output(0)], [model1_p0, model1_p1], 'MergedModel')
    try:
        merged_model.validate_nodes_and_infer_types()
    except Exception:
        pass
    return merged_model

def main():
    bias = const([1, 6, 1, 1])
    weight_1 = const([6, 6, 3, 3])
    weight_2 = const([6, 6, 3, 3])

    model_1 = _model(weight_1, bias)
    model_2 = _model(weight_2, bias)

    merged_model = _merge_model(model_1, model_2)

    input1 = Tensor(np.random.rand(1, 6, 4, 4).astype(np.float32))
    input2 = Tensor(np.random.rand(1, 6).astype(np.float32))

    ov = Core()
    compiled_model = ov.compile_model(merged_model, "CPU")

    infer_request = compiled_model.create_infer_request()
    infer_request.set_input_tensor(0, input1)
    infer_request.set_input_tensor(1, input2)
    infer_request.infer()
    output_1 = infer_request.get_output_tensor(0)

    print("Model 1 output shape:", output_1.shape)
    print("Model 1 output data:", output_1.data[0][0][0])

    # save the merged model for visualization
    serialize(merged_model, "merged_model.xml", "merged_model.bin")

if __name__ == "__main__":
    main()
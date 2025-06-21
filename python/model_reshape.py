
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset1 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model():
    input = opset.parameter([-1, -1, 1, 2], Type.f32, name = 'input')

    output_shape = np.array([0, -1, 2]).astype(np.int32)

    # Refer: https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/shape/reshape-1.html
    # For output_shape, if special_zero=True, 
    # 0: means copy data from original input shape.
    # -1: means need to be calculated.
    op_reshape = opset.reshape(input, output_shape=output_shape, special_zero=True)
 
    Result = opset.result(op_reshape, name='output')
    return Model([Result], [input], 'model_reshape')

def test_reshape():
    core = Core()
    m = model()

    # Save model
    ov.save_model(m, "./tmp_model_reshape.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    # input.shape= (2, 3, 4)
    input = np.random.rand(20, 10, 1, 2).astype(np.float32)

    print("== input.shape=", input.shape)

    ov_result = ireq.infer([input])['output']
    print("---------------------->")
    print("== result.shape=", ov_result.shape)

if __name__ == "__main__":
    test_reshape()
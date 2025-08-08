
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset1 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])
def new_const_dim(val):
    return op.Constant(Type.i64, Shape([len(val)]), val)

def model():
    input = opset.parameter([-1, -1, -1, -1], Type.f32, name = 'input')

    input_f32 = opset.convert(input, Type.f32, "convert")
    op_transpose = opset.transpose(input_f32, new_const_dim([0, 3, 1, 2]), "op_transpose")

    Result = opset.result(op_transpose, name='output')
    return Model([Result], [input], 'model_transpose')

def test_transpose():
    core = Core()
    m = model()

    # Save model
    # ov.save_model(m, "./tmp_model_transpose.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    input = np.random.rand(20, 10, 1, 2).astype(np.int32)

    print("== input.shape=", input.shape)

    ov_result = ireq.infer([input])['output']
    print("---------------------->")
    print("== result.shape=", ov_result.shape)

if __name__ == "__main__":
    test_transpose()
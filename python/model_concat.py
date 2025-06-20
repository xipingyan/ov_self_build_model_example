
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset1 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model(FeaDim):
    input_0 = opset.parameter([-1, 1, FeaDim, 1], Type.f32, name = 'input_0')
    input_1 = opset.parameter([-1, 1, FeaDim, 1], Type.f32, name = 'input_1')
    input_2 = opset.parameter([-1, 1, FeaDim, 1], Type.f32, name = 'input_2')

    op_concat = opset.concat([input_0, input_1, input_2], axis=-1, name="my_concat")
 
    Result = opset.result(op_concat, name='output')
    return Model([Result], [input_0, input_1, input_2], 'model_concat')

def test_concat():
    core = Core()
    FeaDim = 2
    m = model(FeaDim=FeaDim)

    # Save model
    ov.save_model(m, "./tmp_model_concat.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    # input.shape= (2, 3, 4)
    input_0 = np.random.rand(1, 1, FeaDim, 1).astype(np.float32)
    input_1 = np.random.rand(1, 1, FeaDim, 1).astype(np.float32)
    input_2 = np.random.rand(1, 1, FeaDim, 1).astype(np.float32)
    
    print("== input.shape=", input_0.shape, input_1.shape, input_2.shape)
    print("== input 0 =\n", input_0)
    print("== input 1 =\n", input_1)
    print("== input 2 =\n", input_2)

    ov_result = ireq.infer([input_0, input_1, input_2])['output']
    print("---------------------->")
    print("== result.shape=", ov_result.shape)
    print("== OV result =\n", ov_result)

if __name__ == "__main__":
    test_concat()
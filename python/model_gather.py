
import numpy as np

import openvino as ov
from openvino.runtime import Core, Model, Type, Shape, op
from openvino.runtime import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model(FeaDim):
    input_ids = opset.parameter([-1, -1, FeaDim], Type.i32, name = 'input')
    input_ids_shape = opset.shape_of(input_ids)

    indics = opset.gather(input_ids_shape, new_const_1_dim(1), new_const_1_dim(0))

    op_gather = opset.gather(input_ids, indics, new_const_1_dim(1))

    reshape = opset.reshape(op_gather, opset.constant([-1, FeaDim], dtype=ov.Type.i64), special_zero=True)
 
    Result = opset.result(reshape, name='output')
    return Model([Result], [input_ids], 'model_gather')

def test_gather():
    core = Core()
    FeaDim = 3
    m = model(FeaDim=FeaDim)

    # Save model
    ov.save_model(m, "./tmp_model_gather.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    # input = np.array([[1], [4]]).astype(np.int32)
    input = np.array([[[111, 112, 113], [121, 122, 123], [131, 132, 133]], 
                      [[211, 212, 213], [221, 222, 223], [231, 232, 233]]]).astype(np.int32)
    assert(FeaDim == input.shape[-1])
    
    print("== input.shape=", input.shape)
    print("== input[:,-1,:].shape=", input[:,-1, :].shape)
    print("== Expected result = input[:,-1,:] =\n", input[:,-1, :])

    # result = compiled_model(input)[compiled_model.output(0)] # or bellow.
    result = ireq.infer(input)
    print(result)
    # print("---------------------->")
    # print("== result.shape=", result.shape)
    # print("== OV result =\n", result)

if __name__ == "__main__":
    test_gather()
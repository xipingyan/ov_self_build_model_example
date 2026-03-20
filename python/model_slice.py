
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model(FeaDim):
    # shape=[batch, seq, fea]
    input_ids = opset.parameter([-1, -1, FeaDim], Type.i32, name = 'input')

    # Slice last element on axis 0: start=-1, stop=-2 (exclusive), step=-1
    # Slice can keep shape, but `gather` will remove the sliced dimension, so need to reshape it back to keep the same shape as input.
    start = op.Constant(Type.i64, Shape([1]), [-1])
    stop = op.Constant(Type.i64, Shape([1]), [-2])
    step = op.Constant(Type.i64, Shape([1]), [-1])
    axis = op.Constant(Type.i64, Shape([1]), [0])
    name = "slice"
    slice_node = opset.slice(input_ids, start, stop, step, axis, name=name)
 
    Result = opset.result(slice_node, name='output')
    return Model([Result], [input_ids], 'model_slice')

def test_slice():
    core = Core()
    FeaDim = 4
    m = model(FeaDim=FeaDim)

    # Save model
    # ov.save_model(m, "./tmp_model_slice.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    # input.shape= [2, 3, 4]
    input = np.array([[[111, 112, 113, 4], [121, 122, 123, 4], [131, 132, 133, 4]], 
                      [[211, 212, 213, 4], [221, 222, 223, 4], [231, 232, 233, 4]]]).astype(np.int32)
    assert(FeaDim == input.shape[-1])
    
    print("== input.shape=", input.shape)
    print("== input[-1,:,:].shape=", input[-1, :, :].shape)
    # Expected result shape: [1, 3, 4]
    expected_result = input[-1, :, :]
    print("== Expected result = input[-1,:,:] =\n", expected_result)

    # result = compiled_model(input)[compiled_model.output(0)] # or bellow.
    ov_result = ireq.infer(input)['output']
    print("---------------------->")
    print("== result.shape=", ov_result.shape)
    print("== OV result =\n", ov_result)

    print("== Expected vs OV result:", (expected_result-ov_result.tolist() < 0.0001).all())

if __name__ == "__main__":
    test_slice()
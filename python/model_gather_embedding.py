from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np

def model():
    data = op.Constant(np.array([[1,2], [21,22], [31,32], [41,42]]).astype(np.int32))
    indices = opset.parameter([1, 2], Type.i32, name = 'indices')
    axis  = op.Constant(Type.i32, Shape([1]), [0])
    print("data=", data)
    print("axis=", axis)
   
    op_gather = opset.gather(data, indices, axis)
 
    Result = opset.result(op_gather, name='output_gather')
    return Model([Result], [indices], 'model_gather')
def test_gather():
    core = Core()
    
    m = model()
    compiled_model = core.compile_model(model=m, device_name="CPU")

    input=np.array([[1, 2]]).astype(np.float32)
    result = compiled_model(input)[compiled_model.output(0)]

    print("indices=", input)
    print("---------------------->")
    print("result shape=", result.shape)
    print("result=", result)

def model_u8():
    input_ids = opset.parameter([-1, -1], Type.i64, name = 'indices')
    cvt_ids = opset.convert(input_ids, np.int32)

    test_big_shape = 0
    if test_big_shape:
        weight_size = 151936
        weight_dim = 4096
    else:
        weight_size = 4
        weight_dim = 2

    # Weight u8
    if test_big_shape:
        weights = op.Constant(np.random.randint(
            255, size=(weight_size, weight_dim)).astype(np.uint8))
    else:
        weights = op.Constant(
            np.array([[1, 2], [21, 22], [31, 32], [41, 42]]).astype(np.uint8))

    cvt_weights_to_fp16 = opset.convert(weights, np.float16)

    # ZP u8
    if test_big_shape:
        zeropoint = op.Constant(np.random.randint(
            255, size=(weight_size, 1)).astype(np.uint8))
    else:
        zeropoint = op.Constant(
            np.array([[1], [1], [1], [1]]).astype(np.uint8))

    cvt_zp_to_fp16 = opset.convert(zeropoint, np.float16)

    # Scale f16
    if test_big_shape:
        scale = op.Constant(np.random.randint(
            2, size=(weight_size, 1)).astype(np.float16))
    else:
        scale = op.Constant(np.array([[2], [2], [2], [2]]).astype(np.float16))

    subtract = opset.subtract(cvt_weights_to_fp16, cvt_zp_to_fp16)
    multiply = opset.multiply(subtract, scale)

    cvt_mul = opset.convert(multiply, np.float32)

    axis = op.Constant(Type.i32, Shape([1]), [0])

    op_gather = opset.gather(cvt_mul, cvt_ids, axis)

    Result = opset.result(op_gather, name='output_gather')
    return Model([Result], [input_ids], 'model_gather')

def test_gather_embedding():
    print("--->Test: test_gather_embedding")
    core = Core()
    m = model_u8()
    compiled_model = core.compile_model(model=m, device_name="CPU")

    input=np.array([[1], [1]]).astype(np.float32)
    result = compiled_model(input)[compiled_model.output(0)]

    print("indices=", input)
    print("---------------------->")
    print("Real     result=", result, "shape=", result.shape, "dtype=", result.dtype)
    print("Expected result= [[[40. 42.]]]")

# test_gather()
test_gather_embedding()
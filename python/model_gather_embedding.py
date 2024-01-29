from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import time

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

def model_u8(weight_np:np.array):
    input_ids = opset.parameter([-1, -1], Type.i64, name = 'indices')
    cvt_ids = opset.convert(input_ids, np.int32)

    # Weight u8
    weight_size = weight_np.shape[0]
    weights = op.Constant(weight_np)
    cvt_weights_to_fp16 = opset.convert(weights, np.float16)

    # ZP u8
    # zeropoint = op.Constant(np.random.randint(255, size=(weight_size, 1)).astype(np.uint8))
    zeropoint = op.Constant(np.full((weight_size, 1), 1).astype(np.uint8))

    cvt_zp_to_fp16 = opset.convert(zeropoint, np.float16)

    # Scale f16
    # scale = op.Constant(np.random.randint(64, size=(weight_size, 1)).astype(np.float16))
    scale = op.Constant(np.full((weight_size, 1), 2).astype(np.float16))
 
    subtract = opset.subtract(cvt_weights_to_fp16, cvt_zp_to_fp16)
    multiply = opset.multiply(subtract, scale)

    cvt_mul = opset.convert(multiply, np.float32)

    axis = op.Constant(Type.i32, Shape([1]), [0])

    gather = opset.gather(cvt_mul, cvt_ids, axis)

    matmul = opset.matmul(gather, op.Constant(np.full((4096, 1), 1).astype(np.float32)), False, False)

    Result = opset.result(matmul, name='output')
    return Model([Result], [input_ids], 'output')

def get_expected(input, weight_np, zp=1, scale=2, have_matmul=1):
    last_dim = 1 if have_matmul else weight_np.shape[1]
    expected=np.zeros((input.shape[0], input.shape[1], last_dim))
    for b in range(input.shape[0]):
        for s in range(input.shape[1]):
            idx = input[b][s].astype(np.int32)
            if idx < 0:
                idx = idx + weight_np.shape[0]
            tmp = (weight_np[idx].astype(np.float32) - zp)*scale
            if have_matmul:
                expected[b][s] = sum(tmp)
            else:
                expected[b][s] = tmp
    return expected
def print_np(prefix, arr):
    print(prefix, "\n", arr, "shape=", arr.shape, "dtype=", arr.dtype)

def test_one_time(prec, input_shape, weight_np):
    core = Core()

    m = model_u8(weight_np)
    core.set_property("CPU", {hints.inference_precision: prec})

    print("--->inference_precision=", prec, "input_shape=", input_shape)
    t1 = time.time()
    compiled_model = core.compile_model(model=m, device_name="CPU")
    t2 = time.time()
    print("  Compile model:", t2 - t1)

    for i in range(10):
        input=np.random.randint(151935, size=input_shape).astype(np.int64)
        result = compiled_model(input)[compiled_model.output(0)]
    return input, result

def test_gather_embedding():
    print("--->Test: test_gather_embedding")
    print("OV version:", ov.get_version())
    
    weight_np = np.random.randint(255, size=(151936, 4096)).astype(np.uint8)
    # weight_np = np.array([[1,2], [21,22], [31,32], [41,42]]).astype(np.uint8)

    for prec in {Type.f32, Type.bf16}:
        test_one_time(prec, input_shape=(1, 1), weight_np=weight_np)
        test_one_time(prec, input_shape=(1, 32), weight_np=weight_np)
        input, result = test_one_time(prec, input_shape=(1, 1024), weight_np=weight_np)

        # Test accuracy:
        print("------------>Test accuracy, inference_precision=", prec)
        expect = get_expected(input, weight_np)
        if not np.allclose(expect, result):
                print_np("Real result:", result)
                print_np("Expected result:", get_expected(input, weight_np))

# test_gather()
test_gather_embedding()
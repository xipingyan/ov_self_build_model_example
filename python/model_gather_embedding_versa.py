from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset1 as opset
from openvino.runtime import opset8
from openvino.runtime.passes import Manager
import numpy as np
import time
import os

def model_u8_versa(weights_i8, input_dim=1048576):
    input_1 = opset.parameter([1,input_dim], Type.f32, name = 'input_1')
    cvt1 = opset.convert(input_1, np.int32, name = "input_1_cvt")
    squeeze = opset.squeeze(cvt1, opset.constant([0], Type.i64, name="squeeze_axes"), name="squeeze")

    # Weight
    weights_f32 = opset.convert(weights_i8, np.float32, name = "weight_i8_2_f32")
    multiply = opset.multiply(weights_f32, op.Constant(np.full((1), 0.5).astype(np.float32)))

    gather = opset8.gather(multiply, squeeze, op.Constant(Type.i32, Shape([1]), [1]))

    Result = opset.result(gather, name='output')
    return Model([Result], [input_1], 'output')

def print_np(prefix, arr):
    print(prefix, "\n", arr, "shape=", arr.shape, "dtype=", arr.dtype)

def test_one_time(input, weights_i8, dev="CPU", input_dim=10):
    core = Core()
    m = model_u8_versa(weights_i8, input_dim)
    print(f"== dev={dev}")

    t1 = time.time()
    compiled_model = core.compile_model(model=m, device_name=dev)
    t2 = time.time()
    print("== Compile model:", t2 - t1)

    for i in range(1):
        result = compiled_model(input)[compiled_model.output(0)]
    return input, result

# test
# Note: In oder to get 'TEMPLATE' plugin, please cmake -DENABLE_TEMPLATE_REGISTRATION=ON
def test_model():
    input_dim=1048576
    input=np.random.randint(257, size=(1, input_dim)).astype(np.float32)
    print_np("input=", input)
    
    weight_shape = (8, 257)
    weight_data = np.random.randint(255, size=weight_shape).astype(np.int8)
    weights_i8 = opset.constant(weight_data, Type.i8, name = "weight_i8")

    input_ref, result_ref = test_one_time(input, weights_i8, 'TEMPLATE', input_dim)
    # print_np("result_ref=", result_ref)

    input_cpu, result_cpu = test_one_time(input, weights_i8, 'CPU', input_dim)
    # print_np("result_cpu=", result_cpu)

    os.environ["WITH_GC"] = "1"
    input_cpu2, result_cpu2 = test_one_time(input, weights_i8, 'CPU', input_dim)
    # print_np("result_cpu2=", result_cpu2)

    print("=============================\nFinal:\n")
    if not np.allclose(result_ref, result_cpu2):
        print_np("result_cpu=", result_cpu2)
        print_np("result_ref=", result_ref)

test_model()
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
    const_0 = opset.constant([0], Type.i64, name="squeeze_axes")
    squeeze = opset.squeeze(cvt1, const_0, name="squeeze")

    # Weight
    weights_f32 = opset.convert(weights_i8, np.float32, name = "weight_i8_2_f32")
    multiply = opset.multiply(weights_f32, op.Constant(np.full((1), 0.00625154).astype(np.float32)))

    gather = opset8.gather(multiply, squeeze, op.Constant(Type.i32, Shape([1]), [1]))
    org_Transpose_412 = opset.unsqueeze(gather,const_0)
    org_Transpose_20 = opset.unsqueeze(org_Transpose_412, opset.constant([2], Type.i64))
    
    conv_weight_data = np.random.randint(1, size=(128,8,1,500)).astype(np.float32)
    conv_weight = opset.constant(conv_weight_data, Type.f32)
    
    # <data strides="1, 500" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="valid" />
    strides = [1, 500]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [1, 1]
    org_Convolution_38 = opset.convolution(org_Transpose_20, conv_weight, strides, pads_begin, pads_end, dilations, auto_pad="valid")

    Result = opset.result(org_Convolution_38, name='output')
    return Model([Result], [input_1], 'output')

def print_np(prefix, arr):
    print(prefix, "\n", arr, "shape=", arr.shape, "dtype=", arr.dtype)

def test_one_time(input, weights_i8, dev="CPU", input_dim=10):
    core = Core()
    m = model_u8_versa(weights_i8, input_dim)
    print(f"== dev={dev}")
    core.set_property("CPU", {hints.inference_precision: Type.bf16})

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
    if not np.allclose(result_ref, result_cpu):
        print_np("result_cpu=", result_cpu)
        print_np("result_ref=", result_ref)

    if not np.allclose(result_ref, result_cpu2):
        print_np("result_cpu2=", result_cpu2)
        print_np("result_ref=", result_ref)

def run_original_model(input, dev='CPU',prec=None):
    core = Core()
    xml='/mnt/disk1/xiping/models/WW14_static_2024.1.0-14988-4ef40f0f0e7-releases_2024_1/versa/tf/tf_frozen/FP16/INT8/1/ov/optimized/versa.xml'
    m = core.read_model(xml)
    if prec is not None:
        core.set_property("CPU", {hints.inference_precision: prec})

    t1 = time.time()
    compiled_model = core.compile_model(model=m, device_name=dev)
    t2 = time.time()
    print("== Compile model:", t2 - t1)

    result = compiled_model(input)[compiled_model.output(0)]
    return result

def test_original_model():
    input_dim=1048576
    input=np.random.randint(257, size=(1, input_dim)).astype(np.float32)

    result_ref = run_original_model(input, "TEMPLATE")
    
    result_cpu = run_original_model(input, "CPU")

    os.environ["WITH_GC"] = "1"
    result_cpu_bf16 = run_original_model(input, "CPU", Type.bf16)
    result_cpu_f16 = run_original_model(input, "CPU", Type.f16)
    result_cpu_f32 = run_original_model(input, "CPU", Type.f32)

    print_np("result_ref=", result_ref)
    print_np("result_cpu_bf16=", result_cpu_bf16)
    print_np("result_cpu_f16=", result_cpu_f16)
    print_np("result_cpu_f32=", result_cpu_f32)
    print_np("result_cpu=", result_cpu)

    print("== Compare final result:=============")
    if not np.allclose(result_ref, result_cpu):
        print_np("result_cpu=", result_cpu)
        print_np("result_ref=", result_ref)

# test_model()
test_original_model()

# Note: 
# ERROR: 
# Device with "TEMPLATE" name is not registered in the OpenVINO Runtime
# Solution: cmake -DENABLE_TEMPLATE_REGISTRATION=ON ..
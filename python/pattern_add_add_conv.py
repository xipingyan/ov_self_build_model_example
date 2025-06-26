from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import os
 
def const(shape):
    w = np.random.uniform(low=-1, high=1.0, size=shape).astype(np.float32)
    return op.Constant(w)
 
def value(*shape):
    return np.random.rand(*shape).astype(np.float32)
 
def model_add_add_conv():
    input1 = opset.parameter([1, 256, 32, 32], Type.f32, name = 'in1')
    input2 = opset.parameter([1, 256, 32, 32], Type.f32, name = 'in2')

    const1 = const([1, 256, 1, 1])
    weight = const([1024, 256, 1, 1])
   
    add1 = opset.add(input2, const1)
    add2 = opset.add(input1, add1)
 
    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [1, 1]
    conv = opset.convolution(add2, weight, strides, pads_begin, pads_end, dilations)
    
    Result = opset.result(conv, name='output')
    return Model([Result], [input1,input2], 'Model15')

if __name__ == "__main__":
    core = Core()
    m = model_add_add_conv()
    cm_cpu = core.compile_model(model=m, device_name="CPU")
    cm_gpu = core.compile_model(model=m, device_name="GPU")
    
    input1 = value(1, 256, 32, 32)
    input2 = value(1, 256, 32, 32)

    result_cpu = cm_cpu([input1, input2])[0]
    result_gpu = cm_gpu([input1, input2])[0]
    print('result_cpu shape:', result_cpu.shape)
    print('result_gpu shape:', result_gpu.shape)

    print('result_cpu:', result_cpu[0][0][0][:5])
    print('result_gpu:', result_gpu[0][0][0][:5])
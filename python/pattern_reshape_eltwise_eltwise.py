from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import os
from utils.load_dump_file import load_dump_file
import openvino as ov
 
def const(shape):
    w = np.random.uniform(low=-1, high=1.0, size=shape).astype(np.float32)
    return op.Constant(w)
 
def value(*shape):
    return np.random.rand(*shape).astype(np.float32)

# 
def model_reshape_eltwise_eltwise():
    input1 = opset.parameter([-1], Type.f32, name = 'in1')

    output_shape = np.array([0, 1, 1]).astype(np.int32)
    op_reshape = opset.reshape(input1, output_shape=output_shape, special_zero=True)

    bias1 = const([2, 1])
    bias2 = const([256])
   
    op_add1 = opset.add(op_reshape, bias1, auto_broadcast='numpy')
    op_add2 = opset.add(op_reshape, bias2, auto_broadcast='numpy')
 
    op_shape1 = opset.shape_of(op_add1)
    op_shape2 = opset.shape_of(op_add2)
    
    Result1 = opset.result(op_shape1, name='outp1')
    Result2 = opset.result(op_shape2, name='outp2')
    return Model([Result1, Result2], [input1], 'Model')

if __name__ == "__main__":
    print("== OV Version:", ov.get_version())
    core = Core()
    m = model_reshape_eltwise_eltwise()
    input1 = value(3)

    # cm_cpu = core.compile_model(model=m, device_name="CPU")
    cm_gpu = core.compile_model(model=m, device_name="GPU")

    # result_cpu = cm_cpu([input1])[0]
    result_gpu = cm_gpu([input1])
    # print('result_cpu shape:', result_cpu.shape)
    # print('result_gpu shape:', result_gpu.shape)

    # print('result_cpu:', result_cpu)
    print('result_gpu:', result_gpu[0], result_gpu[1])
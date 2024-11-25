from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import os
 
def const(shape):
    w = np.random.rand(*shape).astype(np.float32)
    return op.Constant(w)
 
def value(*shape):
    return np.random.rand(*shape).astype(np.float32)
 
def model():
    input = opset.parameter([-1, 160, -1, -1], Type.f32, name = 'in')
    input2 = opset.parameter([1, 160], Type.f32, name = 'in2')
   
    b2 = opset.reshape(input2, op.Constant(Type.i32, Shape([4]), [1,160,1,1]), special_zero=False)
 
    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    dilations = [1, 1]
    weight = const([160, 160, 3, 3])
    conv = opset.convolution(input, weight, strides, pads_begin, pads_end, dilations)
    conv = opset.add(conv, const([1, 160, 1, 1]))
    conv = opset.add(conv, b2)
    Result = opset.result(conv, name='18/sink_port_0')
    return Model([Result], [input,input2], 'Model15')

ov_device = os.getenv("OV_DEVICE")
if ov_device is None:
    print("== Not set device ENV: OV_DEVICE, default adopt 'CPU'.")
    ov_device = 'CPU'
print("== Test device is: ", ov_device)

core = Core()
m = model()
# Reshape model with range dims.
m.reshape({m.input(0): [1,160,[128,256],[128, 256]]})
cm = core.compile_model(model=m, device_name=ov_device)
 
result = cm([value(1, 160, 128, 128), value(1, 160)])[cm.output(0)]
print('Result shape:', result.shape)
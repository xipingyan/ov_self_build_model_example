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

def new_const_dim(val):
    return op.Constant(Type.i64, Shape([len(val)]), val)

# Model
# input[?,6,?,64]
#     |
# Transpose[?,?,6,64]
#     |
#  Reshape[??,384]     const1[384*384]
#        \              /
#         \            /
#           MatMul[?,?,384]  const2[1,1,384]
#                   \        /
#                   Add[?,?,384]
#                    |
#                 Result
def model():
    input = opset.parameter([-1, 6, -1, 64], Type.f32, name='input')

    transpose = opset.transpose(input, new_const_dim([0, 2, 1, 3]))

    # Bug: only input 1 dynamic shape dim.
    reshape = opset.reshape(transpose, op.Constant(Type.i32, Shape([3]), [1,-1,384]), special_zero = False)

    matmul = opset.matmul(reshape, const([384, 384]), False, True)

    add = opset.add(matmul, const([1, 1, 384]))

    Result = opset.result(add, name='model_mm_result')
    return Model([Result], [input], 'Model_MM')

ov_device = os.getenv("OV_DEVICE")
if ov_device is None:
    print("== Not set device ENV: OV_DEVICE, default adopt 'CPU'.")
    ov_device = 'CPU'
print("== Test device is: ", ov_device)

core = Core()
compiled_model = core.compile_model(model=model(), device_name=ov_device)
 
result = compiled_model([value(1, 6, 1500, 64)])[compiled_model.output(0)]
print('Result shape:', result.shape)
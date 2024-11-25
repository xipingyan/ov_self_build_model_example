from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import os
 
def const(shape, max=1, min=0):
    w = np.random.rand(*shape).astype(np.float32)
    return op.Constant(w)
 
def value(*shape):
    return np.random.rand(*shape).astype(np.float32)

def new_const_dim(val):
    return op.Constant(Type.i64, Shape([len(val)]), val)

def get_const_weight(shape):
    weight_fn = "weight.npy"
    if not os.path.exists(weight_fn):
        w = np.random.rand(*shape).astype(np.float32)
        with open(weight_fn, 'wb') as f:
            np.save(f, w)
            f.close()
    with open(weight_fn, 'rb') as f:
        weights = np.load(f)
        f.close()
    return op.Constant(weights)

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
    # input = opset.parameter([1, 6, -1, 64], Type.f32, name='input')
    input = opset.parameter([1, -1, 2], Type.f32, name='input')

    # transpose = opset.transpose(input, new_const_dim([0, 2, 1, 3]))

    # # Bug: only input 1 dynamic shape dim.
    # reshape = opset.reshape(transpose, op.Constant(
    #     Type.i32, Shape([3]), [1, -1, 384]), special_zero=False)

    matmul = opset.matmul(input, get_const_weight([2, 2]), False, True)

    # add = opset.add(matmul, const([1, 1, 384]))

    Result = opset.result(matmul, name='model_mm_result')
    return Model([Result], [input], 'Model_MM')

ov_device = os.getenv("OV_DEVICE")
if ov_device is None:
    print("== Not set device ENV: OV_DEVICE, default adopt 'CPU'.")
    ov_device = 'CPU'
print("== Test device is: ", ov_device)

core = Core()
compiled_model = core.compile_model(model=model(), device_name=ov_device)
compiled_model_ref = core.compile_model(model=model(), device_name='TEMPLATE')

irq = compiled_model.create_infer_request()
irq_ref = compiled_model_ref.create_infer_request()

input_fn = "input.npy"
if not os.path.exists(input_fn):
    # Range[0,1)
    # input = [value(1, 6, 1500, 64)]
    input = value(1, 5, 2)
    with open(input_fn, 'wb') as f:
        np.save(f, input)
        f.close()
with open(input_fn, 'rb') as f:
    input = np.load(f)
    f.close()

print("== input: ", input)

result = irq.infer(input)[compiled_model.output(0)]
result_ref = irq_ref.infer(input)[compiled_model_ref.output(0)]

print('Result shape:', result.shape)
print('Comare with reference:', 
      np.allclose(result.data.tolist(), result_ref.data.tolist(), 0.001, 0.001, False))
print("result=",result.data.tolist())
print("result_ref=",result_ref.data.tolist())
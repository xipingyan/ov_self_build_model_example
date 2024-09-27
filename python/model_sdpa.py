from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import time
import os

def new_const_dim(val):
    return op.Constant(Type.i64, Shape([len(val)]), val)

def model():
    params = [opset.parameter([-1, -1, 6], Type.f32, name='input0'),
              opset.parameter([-1, -1, 6], Type.f32, name='input1'),
              opset.parameter([-1, -1, 6], Type.f32, name='input2')]

    input_qkv=[]
    for i in range(3):
        shapeof = opset.shape_of(params[i])
        op_gather = opset.gather(shapeof, new_const_dim([0, 1]), new_const_dim([0]))
        reshape_axis = opset.concat([op_gather, new_const_dim([2,3])], axis=0)
        in_reshape = opset.reshape(params[i], reshape_axis, special_zero=0)
        in_transpose = opset.transpose(in_reshape, new_const_dim([0, 2, 1, 3]))
        input_qkv.append(in_transpose)
    
    kwargs = {
        "query": input_qkv[0],
        "key": input_qkv[1],
        "value": input_qkv[2],
    }
    sdpa = ov.opset13.scaled_dot_product_attention(**kwargs)
    out_transpose = opset.transpose(sdpa, new_const_dim([0, 2, 1, 3]))
    out_reshape = opset.reshape(out_transpose, opset.shape_of(params[0]), special_zero=0)

    Result = opset.result(out_reshape, name='output')
    return Model([Result], params, 'model_gather')

def test_gather():
    core = Core()
    m = model()
    compiled_model = core.compile_model(model=m, device_name="CPU")

    input_q = np.array([[[1,2,3,4,5,6]]]).astype(np.float32)
    input_k = np.array([[[1,2,3,4,5,6],[21,22,23,24,25,26]]]).astype(np.float32)
    input_v = np.array([[[1,2,3,4,5,6],[31,32,33,34,35,36]]]).astype(np.float32)
    result = compiled_model([input_q, input_k, input_v])[compiled_model.output(0)]

    print("== input_q=", input_q)
    print("== input_k=", input_k)
    print("== input_v=", input_v)
    print("---------------------->")
    print("== output shape =", result.shape)
    print("== output =", result)

test_gather()
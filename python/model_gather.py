from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import time
import os

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model():
    input_ids = opset.parameter([-1, -1], Type.i32, name = 'input')
    input_ids_shape = opset.shape_of(input_ids)

    # op_gather = opset.gather(input_ids_shape, new_const_1_dim(1), new_const_1_dim(0))

    op_concat = opset.concat([input_ids_shape, new_const_1_dim(6),
                              new_const_1_dim(64)], axis=0)
 
    Result = opset.result(op_concat, name='output')
    return Model([Result], [input_ids], 'model_gather')

def test_gather():
    core = Core()
    m = model()
    compiled_model = core.compile_model(model=m, device_name="CPU")

    input = np.array([[1], [4]]).astype(np.int32)
    result = compiled_model(input)[compiled_model.output(0)]

    print("input=", input)
    print("---------------------->")
    print("result shape=", result.shape)
    print("result=", result)

test_gather()
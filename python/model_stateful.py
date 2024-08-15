# Test stateful model
# Learn nodes: ReadValue, Assign

from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import time
import os

def model_stateful_with_const_input():
    input1 = opset.parameter([1, 4], Type.f32, name = 'data')

    const1 = op.Constant(np.full((4, 1), 2).astype(np.int32))
    convert1 = opset.convert(const1, ov.Type.f32)

    # Stateful
    var_info = op_util.VariableInfo()
    var_info.data_shape = PartialShape([4, 1])
    var_info.data_type = Type.f32
    var_info.variable_id = "v1"
    variable_1 = op_util.Variable(var_info)
    rv = ov.opset6.read_value(convert1, variable_1)
    assign = ov.opset6.assign(rv, variable_1)

    # matmul = opset.matmul(input1, convert1, False, False)
    matmul = opset.matmul(input1, rv, False, False)
    
    res = ov.opset6.result(matmul, "res")
    return Model(results=[res], sinks=[assign], parameters=[input1], name='model_stateful_with_const_input')

def model_stateful_wo_const_input():
    input1 = opset.parameter([1, 4], Type.f32, name = 'input1')
    input2 = opset.parameter([4, 1], Type.i32, name = 'input2')

    convert1 = opset.convert(input2, ov.Type.f32)
    readvalue = opset.read_value(convert1, "rv_1", np.float32)
 
    matmul = opset.matmul(input1, readvalue, False, False)
    
    return Model([matmul], [input1], 'model_stateful_with_const_input')


def test_model_stateful_const_input():
    core = Core()
    model1 = model_stateful_with_const_input()
    compiled_model_1 = core.compile_model(model=model1, device_name="CPU")
    input1=np.array([[1,2,3,4]]).astype(np.float32)

    print("==Run stateful model with const input")
    for i in range(3):
        result = compiled_model_1(input1)[compiled_model_1.output(0)]
        print(f'  loop {i}, reuslt={result}')

def test_model_stateful_input():
    core = Core()
    model2 = model_stateful_wo_const_input()
    compiled_model_2 = core.compile_model(model=model2, device_name="CPU")

    input1=np.array([[1,2,3,4]]).astype(np.float32)
    input2=np.array([[2],[2],[2],[2]]).astype(np.int32)

    print("==Run stateful model without const input")
    for i in range(3):
        result = compiled_model_2([input1, input2])[compiled_model_2.output(0)]
        print(f'  loop {i}, reuslt={result}')

test_model_stateful_const_input()
# test_model_stateful_input()
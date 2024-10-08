# Test ov::Node::If

from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import time
import os

# ======== Model graph ========
#              input1
#                 |
# intput2        If     input3
#    |          /   \      |
#  then(multiply)  else(multiply)
#               \   /
#                Add
#                 |
#               Output
def model_if():
    input1 = opset.parameter([1], Type.i32, "input1")
    input_for_then = opset.parameter([1], Type.i32, "input_for_then")
    input_for_else = opset.parameter([1], Type.i32, "input_for_else")

    # equal = ov.opset8.equal(input1.output(0), op.Constant(np.full((1), 2).astype(np.int32)))
    if_op = ov.opset8.if_op()

    input_then = ov.opset8.parameter([1], np.int32, "input_then")
    multiply1 = opset.multiply(input_then, op.Constant(np.full((1), 2).astype(np.int32)))
    multiply1.set_friendly_name("multiply1")
    then_op_result = ov.opset6.result(multiply1, "res_then")
    body_then_function = Model(results=[then_op_result], parameters=[input_then], name='model_then_body')

    input_else = ov.opset8.parameter([1], np.int32, "input_else")
    multiply2 = opset.multiply(input_else, op.Constant(np.full((1), 4).astype(np.int32)))
    multiply2.set_friendly_name("multiply2")
    else_op_result = ov.opset6.result(multiply2, "res_else")
    body_else_function = Model(results=[else_op_result], parameters=[input_else], name='body_else_function')

    if_op.set_then_body(body_then_function)
    if_op.set_else_body(body_else_function)

    if_op.set_input(input1.output(0), input_then, input_else)
    if_op.set_input(input_for_then.output(0), input_then, None)
    if_op.set_input(input_for_else.output(0), None, input_else)

    result_if = if_op.set_output(then_op_result, else_op_result)

    add = opset.add(result_if, op.Constant(np.full((1), 0).astype(np.int32)))
    res = ov.opset6.result(add, "res")
    return Model(results=[res], parameters=[input1, input_for_then, input_for_else], name='model_if')

def test_model_if(device:str):
    print(f'ov version:{ov.get_version()}')
    core = Core()
    model = model_if()
    compiled_model = core.compile_model(model=model, device_name=device)

    input1=np.array([1]).astype(np.float32)
    input_for_then=np.array([2]).astype(np.int32)
    input_for_else=np.array([4]).astype(np.int32)
    infer_request = compiled_model.create_infer_request()

    print(f"== Run model_if, device={device}")
    result = infer_request.infer([input1, input_for_then, input_for_else])[compiled_model.output(0)]
    print(f'== reuslt={result}')

# == Test ReadValue's input is parameter
# test_model_if('TEMPLATE')
test_model_if('CPU')
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

# ======== Model graph ========
#            const1
#              |
#            convert  
#              |
#   input1  ReadValue
#      \      /
#      Multiply
#         |
#      Output
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

    # Note: MatMul's result is wrong, but Multiply's result is right.
    # m = opset.matmul(input1, convert1, False, False)
    # m = opset.matmul(input1, rv, False, False)
    m = opset.multiply(input1, rv)

    res = ov.opset6.result(m, "res")
    return Model(results=[res], sinks=[assign], parameters=[input1], name='model_stateful_with_const_input')

def test_model_stateful_const_input(device:str):
    core = Core()
    model1 = model_stateful_with_const_input()
    compiled_model_1 = core.compile_model(model=model1, device_name=device)
    irq = compiled_model_1.create_infer_request()

    # states = irq.query_state()
    # target_state = states[0]
    # print(f'states={states[0]}')
    # irq.reset_state()
    
    input1=np.array([[1,2,3,4]]).astype(np.float32)

    print(f"==Run stateful model: ReadValue have const input, device_name={device}")
    for i in range(3):
        result = irq.infer(input1)[compiled_model_1.output(0)]
        print(f'  loop {i}, reuslt={result}')
        # trigger stateful node to init again.
        # irq.reset_state()

# ======== Model graph ========
# input1   intput2
#   \         |
#    \     convert  const1
#     \        \     /
#      \        matmul
#       \         |
#        \     ReadValue
#         \      /
#         Multiply
#            |
#          Output
def model_stateful_with_var_input():
    input1 = opset.parameter([1, 4], Type.f32, name = 'input1')
    input2 = opset.parameter([4, 1], Type.i32, name = 'input2')

    const1 = op.Constant(np.full((1, 1), 2).astype(np.float32))
    convert1 = opset.convert(input2, ov.Type.f32)
    matmul = opset.matmul(convert1, const1, False, False)

    # Stateful
    var_info = op_util.VariableInfo()
    var_info.data_shape = PartialShape([4, 1])
    var_info.data_type = Type.f32
    var_info.variable_id = "v1"
    variable_1 = op_util.Variable(var_info)
    rv = ov.opset6.read_value(matmul, variable_1)
    assign = ov.opset6.assign(rv, variable_1)

    m = opset.multiply(input1, rv)

    res = ov.opset6.result(m, "res")
    return Model(results=[res], sinks=[assign], parameters=[input1, input2], name='model_stateful_with_var_input')

def test_model_stateful_var_input(device:str):
    core = Core()
    model2 = model_stateful_with_var_input()
    compiled_model_2 = core.compile_model(model=model2, device_name=device)

    input1=np.array([[1,2,3,4]]).astype(np.float32)
    input2=np.array([[2],[2],[2],[2]]).astype(np.int32)

    print(f"==Run stateful model: ReadValue have parameter input, device={device}")
    for i in range(3):
        tmp = input2 * i
        # print(f'input2 = {tmp}, i={i}')
        result = compiled_model_2([input1, tmp])[compiled_model_2.output(0)]
        print(f'  loop {i}, reuslt={result}')

        # Trigger ReadValue to init value again.
        compiled_model_2.reset_state()

# == Test ReadValue's input is const
# test_model_stateful_const_input('TEMPLATE')
# test_model_stateful_const_input('CPU')

# == Test ReadValue's input is parameter
test_model_stateful_var_input('TEMPLATE')
test_model_stateful_var_input('CPU')
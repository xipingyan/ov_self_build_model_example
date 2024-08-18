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
#        \     ReadValue  input3
#         \      /         /
#         Multiply  ReadValue
#               \   /
#                Add
#                 |
#               Output
def model_stateful_with_var_input():
    input1 = opset.parameter([1, 4], Type.f32, name = 'input1')
    input2 = opset.parameter([4, 1], Type.i32, name = 'input2')
    input3 = opset.parameter([1], Type.f32, name = 'input3')

    const1 = op.Constant(np.full((1, 1), 2).astype(np.float32))
    convert1 = opset.convert(input2, ov.Type.f32)
    matmul = opset.matmul(convert1, const1, False, False)

    # Stateful rv_2
    var_info = op_util.VariableInfo()
    var_info.data_shape = PartialShape([4, 1])
    var_info.data_type = Type.f32
    var_info.variable_id = "rv_2"
    var_2 = op_util.Variable(var_info)
    rv_2 = ov.opset6.read_value(matmul, var_2)
    assign2 = ov.opset6.assign(rv_2, var_2)

    m = opset.matmul(input1, rv_2, False, False)

    # Stateful rv_3
    var_info3 = op_util.VariableInfo()
    var_info3.data_shape = PartialShape([1])
    var_info3.data_type = Type.f32
    var_info3.variable_id = "rv_3"
    var_3 = op_util.Variable(var_info3)
    rv_3 = ov.opset6.read_value(input3, var_3)
    assign3 = ov.opset6.assign(rv_3, var_3)

    add = opset.add(m, rv_3)

    res = ov.opset6.result(add, "res")
    return Model(results=[res], sinks=[assign2, assign3], parameters=[input1, input2, input3], name='model_stateful_with_var_input')

from openvino.runtime.op.util import VariableInfo, Variable
from openvino.runtime.passes import LowLatency2, MakeStateful, Manager
from openvino.runtime.utils import replace_node
def test_model_stateful_var_input(device:str):
    print(f'ov version:{ov.get_version()}')
    core = Core()
    model2 = model_stateful_with_var_input()
    compiled_model_2 = core.compile_model(model=model2, device_name=device)

    input1=np.array([[1,2,3,4]]).astype(np.float32)
    infer_request = compiled_model_2.create_infer_request()

    print(f"==Run stateful model: ReadValue have parameter input, device={device}")
    for i in range(3):
        input2 = np.array([[i+1], [i+1], [i+1], [i+1]]).astype(np.int32)
        input3 = np.array([i+1]).astype(np.float32)
        # print(f'input2 = {tmp}, i={i}')
        result = infer_request.infer([input1, input2, input3])[
            compiled_model_2.output(0)]
        print(f'  loop {i}, reuslt={result}')

        # Trigger ReadValue to init value again.
        # compiled_model_2.reset_state()
        states = infer_request.query_state()
        for state in states:
            if state.name in ["rv_2", "rv_3"]:
            # if state.name in ["rv_3"]:
                print(f"== state: {state.name}.reset()")
                state.reset()

# == Test ReadValue's input is const
# test_model_stateful_const_input('TEMPLATE')
# test_model_stateful_const_input('CPU')

# == Test ReadValue's input is parameter
# test_model_stateful_var_input('TEMPLATE')
test_model_stateful_var_input('CPU')
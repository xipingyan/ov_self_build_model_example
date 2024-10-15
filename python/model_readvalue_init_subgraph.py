from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
import numpy as np
import time
import os

def rv_construct(inp_node, var_id):
    # Stateful
    var_info = op_util.VariableInfo()
    var_info.data_shape = inp_node.get_input_partial_shape(0)
    var_info.data_type = Type.i32
    var_info.variable_id = var_id
    variable_1 = op_util.Variable(var_info)
    rv = ov.opset6.read_value(inp_node, variable_1)
    assign = ov.opset6.assign(rv, variable_1)
    return rv, assign

# ======== Model graph ========
#         input1    input2
#        /    \     /
#       |  c2 multiply1
#       |   \   /     \
#       |  multiply2  res2
#       |      \
#       |    ReadValue
#        \     /   \
#       multiply3  assign
#           |
#          res1
def model_readvalue():
    input1 = opset.parameter([1], Type.i32)
    input2 = opset.parameter([1], Type.i32)
    multiply1 = opset.multiply(input2, input1)
    const1 = op.Constant(np.full((1), 2).astype(np.int32))
    multiply2 = opset.multiply(multiply1, const1)

    rv, assign = rv_construct(multiply2, var_id="v1")

    multiply3 = opset.multiply(input1, rv)
    res1 = ov.opset6.result(multiply3, "res1")
    res2 = ov.opset6.result(multiply1, "res2")
    return Model(results=[res1, res2], sinks=[assign], parameters=[input1, input2], name='model_readvalue')

def test_model_readvalue_init_subgraph(device:str):
    print(f'ov version:{ov.get_version()}, device={device}')
    core = Core()
    model = model_readvalue()
    compiled_model = core.compile_model(model=model, device_name=device)
    infer_request = compiled_model.create_infer_request()

    reset_id = 3

    for i in range(7):
        input1 = np.array([i+1]).astype(np.int32)
        input2 = np.array([i+1]).astype(np.int32)
        if i == 0:
            state = input1*input2*2

        if i == reset_id:
            # After reset, state should be from input.
            print(f"----> reset_state")
            infer_request.reset_state()
            state = input1*input2*2

        print(f"=====================================")
        expected = input1 * state
        result = infer_request.infer([input1, input2])[compiled_model.output(0)]

        # Assign
        state = state

        is_expected = np.array_equal(result, expected)
        print(f"** Infer:{i+1}, input1={input1}, input2={input2}, is expected={is_expected}")
        if is_expected == False:
            print(f'  --> result={result}, expected={expected}')

# Include: 2 readvalue, 1 static, 1 dynamic
# =============== Model graph =============
#  c2   input2[1]           c2  input3[-1]
#    \    |                   \   /
#    multiply1              multiply2
#        |                      |
#    readvalue1  input1[1]  readvalue2
#      /  \       /   \       /   \
# assign1 multiply3   multiply4  assign2
#                 \   /
#                  add
#                   |
#                  res
def model_2_rv_3_inputs():
    input1 = opset.parameter([1], Type.i32)
    input2 = opset.parameter([1], Type.i32)
    const1 = op.Constant(np.full((1), 2).astype(np.int32))
    multiply1 = opset.multiply(input2, const1)
    rv1, assign1 = rv_construct(multiply1, var_id="v1")

    input3 = opset.parameter([-1], Type.i32)
    const2 = op.Constant(np.full((1), 2).astype(np.int32))
    multiply2 = opset.multiply(input3, const2)
    rv2, assign2 = rv_construct(multiply2, var_id="v2")

    multiply3 = opset.multiply(input1, rv1)
    multiply4 = opset.multiply(input1, rv2)
    add = opset.add(multiply3, multiply4)
    res = ov.opset6.result(add, "res")
    return Model(results=[res], sinks=[assign1, assign2], parameters=[input1, input2, input3], name='model_readvalue_2')

def test_model_2_rv_3_inputs(device:str, optimize=False):
    print(f'ov version:{ov.get_version()}, device={device}')
    core = Core()
    model = model_2_rv_3_inputs()
    compiled_model = core.compile_model(model=model, device_name=device)
    infer_request = compiled_model.create_infer_request()

    reset_id = 3

    print(f"==Start: ===================================")
    for i in range(7):
        input1 = np.array([i+1]).astype(np.int32)
        input2 = np.array([i+1]).astype(np.int32)
        input3 = np.array([i+1]).astype(np.int32)
        if i == 0:
            state_1 = input2 * 2
            state_2 = input3 * 2

        if i == reset_id:
            print(f"----> reset_state")
            infer_request.reset_state()
            # After reset, state should be from input.
            state_1 = input2 * 2
            state_2 = input3 * 2

        result = infer_request.infer([input1, input2, input3])[compiled_model.output(0)]
        expected = (input1 * state_1) + (input1 * state_2)

        # Assign
        state_1 = state_1
        state_2 = state_2

        is_expected = np.array_equal(result, expected)
        print(f"** Infer:{i+1}, input1={input1}, input2={input2}, input3={input3}, is_expected={is_expected}")
        if is_expected is False:
            print(f"  --> result={result}, expected={expected}")

# =====================================
# Start test...
# =====================================
# test_model_readvalue_init_subgraph('TEMPLATE')
test_model_readvalue_init_subgraph('CPU')

# test_model_2_rv_3_inputs('TEMPLATE')
# test_model_2_rv_3_inputs('CPU')
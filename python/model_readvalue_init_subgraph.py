from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
import numpy as np
import time
import os
from colorama import Fore, Back, Style

def print_with_color(var:bool):
    if var:
        return Fore.GREEN, var, Style.RESET_ALL
    else:
        Fore.RED, var, Style.RESET_ALL

def rv_construct(inp_node, var_id, shape):
    # Stateful
    var_info = op_util.VariableInfo()
    var_info.data_shape = shape
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

    rv, assign = rv_construct(multiply2, var_id="v1", shape=multiply2.get_input_partial_shape(0))

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

# ======== Model graph ========
#         input1    const1
#            \     /
#          convolution1
#               |
#   input2  ReadValue
#        \     /   \
#    convolution2  assign
#          |
#         res1
def model_readvalue_no_inplace():
    input1 = opset.parameter([1, 1, 2, 2], Type.i32)
    input2 = opset.parameter([1, 1, 2, 2], Type.i32)

    conv_weight_data = np.array([[[[2,2],[2,2]]]])
    conv_weight = opset.constant(conv_weight_data, Type.i32)
    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [0, 0]
    dilations = [1, 1]
    convolution1 = opset.convolution(input1, conv_weight, strides, pads_begin, pads_end, dilations)

    var_id="v1"
    var_info = op_util.VariableInfo()
    var_info.data_shape = PartialShape([1, 1, 2, 2])
    var_info.data_type = Type.i32
    var_info.variable_id = var_id
    variable_1 = op_util.Variable(var_info)
    rv = ov.opset6.read_value(convolution1, variable_1)
    assign = ov.opset6.assign(rv, variable_1)

    # conv_weight_data = np.array([[[[2,2],[2,2]]]])
    # conv_weight2 = opset.constant(conv_weight_data, Type.i32)
    # strides = [1, 1]
    # pads_begin = [1, 1]
    # pads_end = [0, 0]
    # dilations = [1, 1]
    convolution2 = opset.convolution(input2, rv, strides, pads_begin, pads_end, dilations)
    # convolution2 = opset.add(rv, op.Constant(np.full((1,1,1,1), 1).astype(np.int32)))
    res1 = ov.opset6.result(convolution2, "res1")

    return Model(results=[res1], sinks=[assign], parameters=[input1, input2], name='model_readvalue_no_inplace')

def test_model_readvalue_no_inplace():
    devices = ['TEMPLATE', 'CPU']
    ref_result=[]
    for device in devices:
        print(f'ov version:{ov.get_version()}, device={device}')
        core = Core()
        model = model_readvalue_no_inplace()
        compiled_model = core.compile_model(model=model, device_name=device)
        infer_request = compiled_model.create_infer_request()

        reset_id = 3
        for i in range(7):
            idx = i + 1
            input1 = np.array([[[[idx, idx], [idx, idx]]]]).astype(np.int32)
            input2 = np.array([[[[idx, idx], [idx, idx]]]]).astype(np.int32)
            if i == reset_id:
                # After reset, state should be from input.
                print(f"----> reset_state")
                infer_request.reset_state()

            print(f"=====================================")
            result = infer_request.infer([input1, input2])[compiled_model.output(0)]

            if device == 'TEMPLATE':
                ref_result.append(result)
            else:
                expected = ref_result[i]
                is_expected = np.array_equal(result, expected)
                print(f"** Infer:{idx}, input1={input1}\n  Result is expected={(Fore.GREEN if is_expected else Fore.RED) + str(is_expected) + Style.RESET_ALL}")
                if is_expected == False:
                    print(f'  ------> Result != Reference\n  result={result}, \n  expected={expected}')

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
    rv1, assign1 = rv_construct(multiply1, var_id="v1", shape=multiply1.get_input_partial_shape(0))

    input3 = opset.parameter([-1], Type.i32)
    const2 = op.Constant(np.full((1), 2).astype(np.int32))
    multiply2 = opset.multiply(input3, const2)
    rv2, assign2 = rv_construct(multiply2, var_id="v2", shape=multiply2.get_input_partial_shape(0))

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

# ReadValue Assign direct pair
#
#             input_1   input_2
#                |        |
#              Add_1     /
#                \      /
#                 MatMul
#                   |
#   input_0     ReadValue ..........
#       \      /       \           .
#         Add_0      Assign ........
#          |
#        Result
def model_rv_with_2_inputs():
    #  {{1, -1}, {{1, 2}, {1, 2}, {1, 1}}},
    #     {{2, -1}, {{2, 3}, {2, 2}, {2, 1}}},
    #     {{-1, 2}, {{3, 2}, {2, 2}, {1, 2}}},

    input1 = opset.parameter([1, -1], Type.i32)
    input2 = opset.parameter([2, -1], Type.i32)
    input3 = opset.parameter([-1, 2], Type.i32)
    const1 = op.Constant(np.full((1), 1).astype(np.int32))
    add_1 = opset.add(input2, const1)
    mm_0 = opset.matmul(add_1, input3, transpose_a=False, transpose_b=False)

    print("-------->", type(mm_0.get_input_partial_shape(0)))

    rv_shape = ov.PartialShape([2, 2])
    rv1, assign1 = rv_construct(mm_0, var_id="v1", shape=rv_shape)

    add_0 = opset.add(input1, rv1)
    res = ov.opset6.result(add_0, "res")
    return Model(results=[res], sinks=[assign1], parameters=[input1, input2, input3], name='model_rv_with_2_inputs')

def test_model_rv_with_2_inputs(device:str, optimize=False):
    print(f'ov version:{ov.get_version()}, device={device}')
    core = Core()
    model = model_rv_with_2_inputs()
    compiled_model = core.compile_model(model=model, device_name=device)
    infer_request = compiled_model.create_infer_request()

    reset_id = 3

    print(f"==Start: ===================================")
    for i in range(7):
        idx = i + 1
        input1 = np.array([[idx, idx]]).astype(np.int32) # [1,2]
        input2 = np.array([[idx, idx, idx],[idx, idx, idx]]).astype(np.int32) # [2,3]
        input3 = np.array([[idx, idx], [idx, idx], [idx, idx]]).astype(np.int32) # [3,2]
        if i == 0:
            state_1 = np.matmul((input2 + 1), input3)

        if i == reset_id:
            print(f"----> reset_state")
            infer_request.reset_state()
            # After reset, state should be from input.
            state_1 = np.matmul((input2 + 1), input3)

        result = infer_request.infer([input1, input2, input3])[compiled_model.output(0)]
        expected = (input1 + state_1)

        # Assign
        state_1 = state_1

        is_expected = np.array_equal(result, expected)
        print(f"** Infer:{i+1}, input1={input1}, input2={input2}, input3={input3}, is_expected={is_expected}")
        if is_expected is False:
            print(f"  --> result={result}, expected={expected}")

# =====================================
# Start test...
# =====================================
# test_model_readvalue_init_subgraph('TEMPLATE')
# test_model_readvalue_init_subgraph('CPU')

# test_model_readvalue_no_inplace('TEMPLATE')
# test_model_readvalue_no_inplace()

# test_model_2_rv_3_inputs('TEMPLATE')
# test_model_2_rv_3_inputs('CPU')

# test_model_rv_with_2_inputs('TEMPLATE')
test_model_rv_with_2_inputs('CPU')
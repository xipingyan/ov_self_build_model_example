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
#         input1  input2
#        /   \      /
#        |   multiply1
#        |      |       \
#        |   multiply   res2
#        |      |
#        |   ReadValue
#        \     /
#       multiply2
#           |
#          res1
def model_readvalue():
    input1 = opset.parameter([1], Type.i32)
    input2 = opset.parameter([1], Type.i32)
    multiply1 = opset.multiply(input2, input1)
    const1 = op.Constant(np.full((1), 2).astype(np.int32))
    multiply = opset.multiply(multiply1, const1)

    rv, assign = rv_construct(multiply, var_id="v1")

    multiply2 = opset.multiply(input1, rv)
    res1 = ov.opset6.result(multiply2, "res1")
    res2 = ov.opset6.result(multiply1, "res2")
    return Model(results=[res1, res2], sinks=[assign], parameters=[input1, input2], name='model_readvalue')

# Include: 2 readvalue, 1 static, 1 dynamic
# =============== Model graph =============
#     input2[1]              input3[-1]
#        |                      |
#     multiply1              multiply2
#        |                      |
#    readvalue1  input1[1]  readvalue2
#         \       /   \       /
#         multiply3   multiply4
#                 \   /
#                  add
#                   |
#                  res
def model_readvalue_2():
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

def model_readvalue_optimize_with_if():
    input1 = opset.parameter([1], Type.i32)
    input2 = opset.parameter([1], Type.i32)

    equal1 = ov.opset8.equal(input1.output(0), op.Constant(np.full((1), 1).astype(np.int32)))
    equal2 = ov.opset8.equal(input1.output(0), op.Constant(np.full((1), 3).astype(np.int32)))
    is_1_or_3 = ov.opset8.greater(ov.opset8.add(ov.opset8.convert(equal1, Type.i32), ov.opset8.convert(equal2, Type.i32)), op.Constant(np.full((1), 0).astype(np.int32)))
    cvted_is_1_or_3 = ov.opset1.convert(is_1_or_3, Type.i32)

    if_op = ov.opset8.if_op()
    # then branch
    # ==================================================
    input_then = ov.opset8.parameter([1], np.int32, "input_then")
    const1 = op.Constant(np.full((1), 2).astype(np.int32))
    multiply = opset.multiply(input_then, const1)
    rv_then, assign_then = rv_construct(multiply, var_id="v1")
    then_op_result = ov.opset6.result(rv_then, "res_then")
    body_then_function = Model(results=[then_op_result], sinks=[assign_then], parameters=[input_then], name='model_then_body')

    # else branch
    # ==================================================
    input_else_const = op.Constant(np.full((1), 0).astype(np.int32))
    rv_else, assign_else = rv_construct(input_else_const, var_id="v2")
    else_op_result = ov.opset6.result(rv_else, "res_else")
    body_else_function = Model(results=[else_op_result], sinks=[assign_else], parameters=[], name='body_else_function')

    # if: input, output, body
    # ==================================================
    if_op.set_then_body(body_then_function)
    if_op.set_else_body(body_else_function)
    if_op.set_input(cvted_is_1_or_3.output(0), input_then, None)
    if_op.set_input(input2.output(0), input_then, None)
    result_if = if_op.set_output(then_op_result, else_op_result)

    m = opset.multiply(input1, result_if)

    res = ov.opset6.result(m, "res")
    return Model(results=[res], parameters=[input1, input2], name='model_if_readvalue')

def test_model_if_readvalue(device:str, optimize=False):
    print(f'ov version:{ov.get_version()}, device={device}')
    core = Core()
    model = model_readvalue_optimize_with_if() if optimize else model_readvalue()
    compiled_model = core.compile_model(model=model, device_name=device)
    infer_request = compiled_model.create_infer_request()

    scale = (0+1) * 2
    for i in range(10):
        input1 = np.array([i+1]).astype(np.int32)
        input2 = np.array([i+1]).astype(np.int32)
        if i == 2:
            infer_request.reset_state()
            scale = (i+1) * 2
        print(f"=====================================")
        print(f"== Infer:{i+1}, input1={input1}, input2={input2}, expected={(i+1)*scale}")
        result = infer_request.infer([input1, input2])[compiled_model.output(0)]
        print(f'== reuslt:{i+1} = {result}')

def test_model_readvalue_2(device:str, optimize=False):
    print(f'ov version:{ov.get_version()}, device={device}')
    core = Core()
    model = model_readvalue_2()
    compiled_model = core.compile_model(model=model, device_name=device)
    infer_request = compiled_model.create_infer_request()

    scale = (0+1) * 2
    for i in range(10):
        input1 = np.array([i+1]).astype(np.int32)
        input2 = np.array([i+1]).astype(np.int32)
        input3 = np.array([i+1]).astype(np.int32)
        if i == 2:
            print(f"== reset_state")
            infer_request.reset_state()
            scale = (i+1) * 2
        print(f"=====================================")
        print(f"== Infer:{i+1}, input1={input1}, input2={input2}, input3={input3}, expected={(i+1)*scale*2}")
        result = infer_request.infer([input1, input2, input3])[compiled_model.output(0)]
        print(f'== reuslt:{i+1} = {result}')
# =====================================
# test_model_if_readvalue('TEMPLATE')
# test_model_if_readvalue('CPU', optimize=False)
# test_model_if_readvalue('CPU', optimize=True)

test_model_readvalue_2('CPU')
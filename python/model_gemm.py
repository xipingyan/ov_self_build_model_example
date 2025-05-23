from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import openvino.runtime as ov
import numpy as np
import os


def const(shape):
    # default generate data with range [0, 1)
    w = np.random.rand(*shape).astype(np.float32)
    return op.Constant(w)

def value(*shape):
    return np.random.rand(*shape).astype(np.float32)

def new_const_dim(val):
    return op.Constant(Type.i64, Shape([len(val)]), val)

# Model
# input[?,?,896]
#     |
#  Reorder[??,896]     const1[151936*896]
#        \              /
#         \            /
#           MatMul[?,?,151936]  (transpose_a="false" transpose_b="true")
#               |
#             Result


Weight_y=896
Weight_x=151936

def model_gemm(weights):
    input = opset.parameter([-1, -1, Weight_y], Type.f32, name='input')

    matmul = opset.matmul(input, weights, False, True)

    Result = opset.result(matmul, name='model_gemm_result')

    return Model([Result], [input], 'Model_MM')

def prepare_weights(weight_shape):
    weight_fn = "weight.npy"
    if os.path.exists(weight_fn):
        with open(weight_fn, 'rb') as f:
            weights = np.load(f)
            f.close()

        if list(np.shape(weights)) == weight_shape:
            print(" == cached weight and required weight shape: ", list(np.shape(weights)), "and", weight_shape, "is same, cached weight is used.")
            return op.Constant(weights)

    # Generate new weights ans save
    w = np.random.rand(*weight_shape).astype(np.float32)
    with open(weight_fn, 'wb') as f:
        np.save(f, w)
        f.close()
    return op.Constant(w)

def prepare_input(input_shape):
    input_fn = "input.npy"
    if os.path.exists(input_fn):
        with open(input_fn, 'rb') as f:
            input = np.load(f)
            f.close()

        if list(np.shape(input)) == input_shape:
            print(" == cached input and required input shape: ", list(np.shape(input)), "and", input_shape, "is same, cached weight is used.")
            return input

    # Generate new weights ans save
    new_input = np.random.rand(*input_shape).astype(np.float32)
    with open(input_fn, 'wb') as f:
        np.save(f, new_input)
        f.close()
    return new_input

def main():
    print("== OpenVINO Version:", ov.get_version())
    ov_device = os.getenv("OV_DEVICE")
    if ov_device is None:
        print("== Not set device ENV: OV_DEVICE, default adopt 'CPU'.")
        ov_device = 'GPU'
    print("== Test device is: ", ov_device)

    test_acc = os.getenv("ACC")
    test_acc = 0 if test_acc is None else 1
    print("== test_acc = ", test_acc)
    
    run_template = False
    print("== run_template: ", run_template)

    # MatMul's weights.
    weights = prepare_weights([Weight_x, Weight_y])

    core = Core()
    model = model_gemm(weights)
    compiled_model = core.compile_model(model=model, device_name=ov_device)
    if run_template:
        compiled_model_ref = core.compile_model(model=model, device_name='TEMPLATE')

    irq = compiled_model.create_infer_request()
    if run_template:
        irq_ref = compiled_model_ref.create_infer_request()

    # Dump execution graph
    runtime_model = compiled_model.get_runtime_model()
    if os.getenv('dump_runtime_model') is not None:
        serialize(runtime_model, "gpu_runtime_graph.xml")

    # Ready input:
    input = prepare_input([1, 8, Weight_y])
    # print("== input: ", input)
    if (test_acc):
        result = irq.infer(input)[compiled_model.output(0)]
    else:
        for i in range(100):
            irq.infer(input)
        result = irq.infer(input)[compiled_model.output(0)]

    print('== Result shape:', result.shape)
    print('== Result.data.tolist()[0][0][0:5]:', result.data.tolist()[0][0][0:10])

    if run_template:
        result_ref = irq_ref.infer(input)[compiled_model_ref.output(0)]

        is_same = np.allclose(result.data.tolist(), result_ref.data.tolist(), 0.001, 0.001, False)
        print('== Result and Reference are',
            'same. Success.' if is_same else 'different. Fail.')
        if is_same is False:
            print("  == result     data =", np.round(result.data.tolist()[0][0][0:10], 4))
            print("  == result_ref data =", np.round(result_ref.data.tolist()[0][0][0:10], 4))
            
            print("  == result    [0] =", type(result.data.tolist()[0][0][0]))
            print("  == result_ref[0] =", type(result_ref.data.tolist()[0][0][0]))


main()

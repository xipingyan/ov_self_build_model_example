from openvino.runtime import Core, Model, Type, Shape, op, serialize
from openvino.runtime import opset8 as opset
import openvino.runtime as ov
import numpy as np
import os

def new_const_dim(val):
    return op.Constant(Type.i64, Shape([len(val)]), val)

# Model
# input[?,6,?,64]
#     |
# Transpose[?,?,6,64]
#     |
#  Reshape[??,384]     const1[384*384]
#        \              /
#         \            /
#           MatMul[?,?,384]  const2[1,1,384]
#                   \        /
#                   Add[?,?,384]
#                    |
#                 Result

Weight_y=192
Weight_x=384
# Weight_Size=2
# Weight_Size=16

def model_mm(weights):
    input = opset.parameter([1, -1, Weight_x], Type.f32, name='input')
    matmul = opset.matmul(input, weights, False, True)

    new_shape = np.array([1, 0, 16, 12], dtype=np.int32)
    reshape = opset.reshape(matmul, new_shape, True)
    # filter weights should have shape M x C x kH x kW
    input_filter = np.array([[[[0.1, 0.2, 0.3],[0.1, 0.2, 0.3],[0.1, 0.2, 0.3]]]], dtype=np.float32)
    conv = opset.convolution(reshape, input_filter, np.array([1, 1]), np.array(
        [1, 1]), np.array([1, 1]), np.array([1, 1]), name="my_convolution")
    Result = opset.result(conv, name='model_mm_result')

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
        print("== Not set device ENV: OV_DEVICE, default adopt 'GPU'.")
        ov_device = 'GPU'
    print("== Test device is: ", ov_device)

    run_template = False
    print("== run_template: ", run_template)

    # MatMul's weights.
    weights = prepare_weights([Weight_y, Weight_x])

    core = Core()
    model = model_mm(weights)

    # serialize(model, "model_old.xml")
    print("== All layers:")
    for op in model.get_ops():
        # print(f"    {op.get_type_name()}[{op.get_friendly_name()}]")
        if op.get_type_name() == "MatMul":
            # print(f"        Add rt info to {op.get_name()}")
            # Add runtime info: DisableFP16Compression for MatMul
            op.rt_info['precise_0'] = ''

    # print("== Print rt info:")
    # for idx, op in enumerate(model.get_ops()):
    #     # if op.get_type_name() in ["MatMul", "Convert"]:
    #     print(f"    {op.get_name()}: {[n for n in op.get_rt_info()]}")
    # serialize(model, "model_mm_f32.xml")

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
    input = prepare_input([1, 1, Weight_x])
    # print("== input: ", input)

    # for outp in compiled_model.outputs():
    #     print("  output=", outp)
    result = irq.infer(input)[compiled_model.output(0)]

    print('===========================================')
    print('== Result shape:', result.shape)
    print('  Real Result[0:5]:', np.round(result.data.tolist()[0][0][0][0:10], 6))

    if run_template:
        result_ref = irq_ref.infer(input)[compiled_model_ref.output(0)]

        is_same = np.allclose(result.data.tolist(), result_ref.data.tolist(), 0.001, 0.001, False)
        print('== Result and Reference are',
            'same. Success.' if is_same else 'different. Fail.')
        if is_same is False:
            print("  == result     data =", np.round(result.data.tolist()[0][0][0][0:10], 4))
            print("  == result_ref data =", np.round(result_ref.data.tolist()[0][0][0][0:10], 4))
            
            print("  == result    [0] =", type(result.data.tolist()[0][0][0][0]))
            print("  == result_ref[0] =", type(result_ref.data.tolist()[0][0][0][0]))

if __name__ == "__main__":
    main()

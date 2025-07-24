from openvino.runtime import Core, Model, Type, serialize
from openvino.runtime import opset8 as opset
import openvino.runtime as ov
import numpy as np
import os

def model_mvn(input_shape):
    input = opset.parameter(input_shape, Type.f32, name='input0')

    axes_node = opset.constant([2], dtype=ov.Type.i64, name="mvn_axes")
    normalize_variance = True
    default_eps = 1e-5
    mvn_node = opset.mvn(input, axes_node, normalize_variance, default_eps, eps_mode='inside_sqrt')

    return Model([mvn_node], [input], 'Model_Mvn')

def prepare_input(input_shape):
    input_fn = "input.npy"
    if os.path.exists(input_fn):
        with open(input_fn, 'rb') as f:
            input = np.load(f)
            f.close()

        if list(np.shape(input)) == input_shape:
            print("  == cached input and required input shape: ", list(np.shape(input)), "and", input_shape, "is same, cached weight is used.")
            return input

    # Generate new weights ans save
    new_input = np.random.rand(*input_shape).astype(np.float32)
    with open(input_fn, 'wb') as f:
        np.save(f, new_input)
        f.close()
    return new_input  

def test_mvn(model_type="static"):
    print("== OpenVINO Version:", ov.get_version())
    ov_device = os.getenv("OV_DEVICE")
    if ov_device is None:
        print("== Not set device ENV: OV_DEVICE, default adopt 'GPU'.")
        ov_device = 'GPU'
    print("== Test device is: ", ov_device)

    run_template = False
    print("== run_template: ", run_template)

    core = Core()
    input_shape = [2,3,4]
    model = model_mvn(input_shape if model_type=="static" else [-1,-1,4])
    compiled_model = core.compile_model(model=model, device_name=ov_device)
    if run_template:
        compiled_model_ref = core.compile_model(model=model, device_name='TEMPLATE')

    irq = compiled_model.create_infer_request()
    if run_template:
        irq_ref = compiled_model_ref.create_infer_request()

    # Dump execution graph
    runtime_model = compiled_model.get_runtime_model()
    if os.getenv('dump_runtime_model') is not None:
        serialize(runtime_model, f"gpu_runtime_graph_{model_type}.xml")

    # Ready input:
    input = prepare_input(input_shape)
    # print("== input: ", input)

    # for outp in compiled_model.outputs():
    #     print("  output=", outp)

    if model_type=="dynamic":
        inputs_dyn = [
            prepare_input([1,2,4]), 
            prepare_input([2,2,4]),
            prepare_input([2,2,4]),
            prepare_input([3,2,4])
        ]

    print("== Start to infer...")
    for i in range(4):
        print(f"  ** Infer [{i}] model_type = {model_type} ...")
        if model_type=="dynamic":
            input = inputs_dyn[i]
        result = irq.infer(input)[compiled_model.output(0)]

    print('===========================================')
    print('== Result shape:', result.shape)
    print('   Real Result[0][0][:] = ', np.round(result.data.tolist()[0][0][:], 4))

    if run_template:
        result_ref = irq_ref.infer(input)[compiled_model_ref.output(0)]
        is_same = np.allclose(result.data.tolist(), result_ref.data.tolist(), 0.001, 0.001, False)
        print(f"== Result and Reference are { 'same. Success.' if is_same else 'different. Fail.'}")

if __name__ == "__main__":
    test_mvn(model_type='static')
    # test_mvn(model_type="dynamic")

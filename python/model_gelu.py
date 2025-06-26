
import numpy as np
import os
import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model():
    input = opset.parameter([1, 1024, 32, 32], Type.f32, name='input')

    op_gelu = opset.gelu(input, approximation_mode="ERF")
 
    Result = opset.result(op_gelu, name='output')
    return Model([Result], [input], 'model_gelu')

def test_gelu():
    core = Core()
    m = model()

    # Save model
    # ov.save_model(m, "./tmp_model_gelu.xml")

    cm_cpu = core.compile_model(model=m, device_name="CPU")
    cm_gpu = core.compile_model(model=m, device_name="GPU")
    ireq_cpu = cm_cpu.create_infer_request()
    ireq_gpu = cm_gpu.create_infer_request()

    input_fn = "./tmp_inpput.npy"
    if os.path.exists(input_fn):
        input = np.load(input_fn)
    else:
        input = np.random.uniform(low=-20, high=20, size=[1,1024,32,32]).astype(np.float32)
        np.save(input_fn, input)

    print("== input.shape=", input.shape)
    print("== input=", input[0][-1][-1][:5])

    ov_result_cpu = ireq_cpu.infer(input)['output']
    ov_result_gpu = ireq_gpu.infer(input)['output']

    print("---------------------->")
    print("== result.shape=", ov_result_cpu.shape, ov_result_gpu.shape)
    print("== ov_result_cpu =", ov_result_cpu.tolist()[0][-1][-1][:5])
    print("== ov_result_gpu =", ov_result_gpu.tolist()[0][-1][-1][:5])


if __name__ == "__main__":
    test_gelu()
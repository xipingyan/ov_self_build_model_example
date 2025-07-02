import openvino as ov
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime import opset8 as opset
import numpy as np
import os
 
def value(*shape):
    return np.random.rand(*shape).astype(np.float32)
 
def model_gather_add():
    input1 = opset.parameter([-1, 2, 3], Type.f32, name = 'in1')
   
    gather = opset.gather(input1, 0, 1)

    bias_weight = opset.constant([[0.1, 0.2, 0.3]], np.float32)
    add1 = opset.add(gather, bias_weight)
 
    Result = opset.result(add1, name='output')
    return Model([Result], [input1], 'Model15')

if __name__ == "__main__":
    core = Core()
    m = model_gather_add()
    ov.save_model(m, "./tmp.xml")

    inp1_fn = "input1.npy"
    if os.path.exists(inp1_fn):
        input1 = np.load(inp1_fn)
    else:
        input1 = value(4, 2, 3)
        np.save(inp1_fn, input1)

    cm_cpu = core.compile_model(model=m, device_name="CPU")
    cm_gpu = core.compile_model(model=m, device_name="GPU")

    result_cpu = cm_cpu([input1])[0]
    result_gpu = cm_gpu([input1])[0]
    print('result_cpu shape:', result_cpu.shape)
    print('result_gpu shape:', result_gpu.shape)

    # print('result_cpu:', result_cpu[:][:])
    # print('result_gpu:', result_gpu[:][:])
    print("Compare CPU and GPU result: ", np.isclose(result_cpu, result_gpu, rtol=0.001, atol=0.001).all())
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

def model_multiply():
    input = opset.parameter([-1, -1, -1, -1], Type.f32, name='input')

    scale = op.Constant(Type.f32, Shape([1, 3, 1, 1]), [1.0, 1.0, 1.0])
    matmul = opset.multiply(input, scale)

    Result = opset.result(matmul, name='model_multiply_result')

    return Model([Result], [input], 'Model_multiply')

def main():
    print("== OpenVINO Version:", ov.get_version())
    ov_device = os.getenv("OV_DEVICE")
    if ov_device is None:
        print("== Not set device ENV: OV_DEVICE, default adopt 'CPU'.")
        ov_device = 'CPU'
    print("== Test device is: ", ov_device)

    core = Core()
    model = model_multiply()
    compiled_model = core.compile_model(model=model, device_name=ov_device)
    irq = compiled_model.create_infer_request()

    # Ready input:
    input = prepare_input([1, 3, 2, 2])
    # print("== input: ", input)

    result = irq.infer(input)[compiled_model.output(0)]

    print('===========================================')
    print('== Result shape:', result.shape)
    print('   All result:', np.round(result.data.tolist(), 4))

if __name__ == "__main__":
    main()

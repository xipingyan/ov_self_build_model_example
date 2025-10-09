from openvino.runtime import opset8 as opset
import openvino as ov
from openvino.runtime import Core, Type, op, Shape, Model
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

def prepare_input(input_shape, type=np.float32):
    input_fn = "input.npy"
    if os.path.exists(input_fn):
        with open(input_fn, 'rb') as f:
            input = np.load(f)
            f.close()

        if list(np.shape(input)) == input_shape:
            print(" == cached input and required input shape: ", list(np.shape(input)), "and", input_shape, "is same, cached weight is used.")
            return input

    # Generate new weights ans save
    new_input = np.random.rand(*input_shape)*255.0
    if type is np.uint8:
        new_input = new_input.astype(type)

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

def model_multiply_complex_case():
    raw_images_1 = opset.parameter([-1, -1, -1, -1], Type.u8, name='input1')
    raw_images_2 = opset.parameter([-1, -1, -1, -1], Type.u8, name='input2')
    resize_shape = opset.parameter([2], Type.i64, name='input3')

    mean = op.Constant(Type.f32, Shape([1, 3, 1, 1]), [0.0, 0., 0.])
    scale = op.Constant(Type.f32, Shape([1, 3, 1, 1]), [1.0, 1.0, 1.0])

    raw_images_1_cvt = opset.convert(raw_images_1, Type.f32)
    img_trans_1 = opset.transpose(raw_images_1_cvt, op.Constant(Type.i32, Shape([4]), [0, 3, 1, 2]))
    clamp_1 = opset.clamp(img_trans_1, 0, 255)
    substract_1 = opset.subtract(clamp_1, mean)
    multiply_1 = opset.multiply(substract_1, scale)

    raw_images_2_cvt = opset.convert(raw_images_2, Type.f32)
    img_trans_2 = opset.transpose(raw_images_2_cvt, op.Constant(Type.i32, Shape([4]), [0, 3, 1, 2]))
    clamp_2 = opset.clamp(img_trans_2, 0, 255)
    substract_2 = opset.subtract(clamp_2, mean)
    multiply_2 = opset.multiply(substract_2, scale)

    concat = opset.concat({multiply_1, multiply_2}, axis=0)

    result_1 = opset.result(concat, name='multiply_1')
    return Model([result_1], [raw_images_1, raw_images_2], 'Model_multiply')

def main():
    print("== OpenVINO Version:", ov.get_version())
    ov_device = os.getenv("OV_DEVICE")
    if ov_device is None:
        print("== Not set device ENV: OV_DEVICE, default adopt 'CPU'.")
        ov_device = 'CPU'
    print("== Test device is: ", ov_device)

    core = Core()
    # model = model_multiply()
    model = model_multiply_complex_case()
    compiled_model = core.compile_model(model=model, device_name=ov_device)
    irq = compiled_model.create_infer_request()

    # Ready input:
    # input = prepare_input([1, 3, 2, 2])
    input = prepare_input([1, 2, 2, 3], type=np.uint8)
    # print("== input: ", input)

    result = irq.infer([input, input])[compiled_model.output(0)]

    print('===========================================')
    print('== Result shape:', result.shape)
    print('   All result:', np.round(result.data.tolist(), 4))

if __name__ == "__main__":
    main()

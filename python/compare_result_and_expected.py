import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset8 as opset

print("OpenVINO Version:", ov.get_version())

src_data = [69,  65,  41,  101, 135, 97, 129, 97,  41, 41,  37,  129, 69, 97,  37,  33,
            5,   37,  73,  101, 73,  67, 33,  69,  39, 103, 135, 37,  3,  69,  101, 35,
            105, 65,  99,  129, 73,  3,  129, 137, 99, 33,  39,  37,  69, 131, 37,  133,
            105, 133, 101, 41,  97,  9,  39,  133, 5,  39,  9,   105, 5,  135, 103, 3]

expected = [5, 4, 1, 4,-7,2,5,6,7,-8,1,6,1,-8,1,6,-7,2,-7,2,5,2,1,-8,5,4,1,6,5,2,1,2,5,
            0,5,2,-7,4,5,6,-7,4,3,4,1,2,5,4,7,2,7,6,7,-8,5,2,3,0,5,4,5,6,3,2,-7,6,1,4,3,
            6,1,-8,-7,4,3,0,1,-8,-7,-8,3,6,1,2,7,2,5,2,5,4,3,-8,5,2,5,-8,-7,6,5,-8,5,6,
            -7,2,1,6,-7,0,7,2,5,-8,5,0,7,2,-7,0,-7,6,5,0,7,-8,7,6,3,0]

def get_input_tensor():
    arr = np.ndarray((64))
    for i in range(len(src_data)):
        arr.data[i] = float(src_data[i])
    input_tensor = ov.Tensor(arr.astype(np.uint8), [128], ov.Type.i4)
    return input_tensor

# Model: it only contains 1 node: convert
def model():
    input = opset.parameter([128], ov.Type.i4, name='in')

    cvt = opset.convert(input, ov.Type.f32)
    Result = opset.result(cvt, name='cvt')
    return ov.Model([Result], [input], 'MyModel')

# Ov model and inference
core = ov.Core()
m = model()
cm = core.compile_model(model=m, device_name="CPU")
input = get_input_tensor()
result = cm([input])[cm.output(0)]

# Convert result to list
result = result.data.tolist()

print(f'Input tensor:{src_data}')
print(f'Output tensor:{result}')

print("Start Compare ---------------------")
assert(len(result) == len(expected))
is_same = True
for i in range(len(result)):
    if result[i] != expected[i]:
        print("diff", i)
        is_same = False
print("Done: ", "Result = Expected" if is_same else "Result != Expected")

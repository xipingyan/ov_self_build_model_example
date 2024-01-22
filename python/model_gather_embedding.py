from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np

def model():
    data = op.Constant(np.array([[1,2], [21,22], [31,32], [41,42]]).astype(np.int32))
    indices = opset.parameter([1, 1], Type.i32, name = 'indices')
    axis  = op.Constant(Type.i32, Shape([1]), [0])
    print("data=", data)
    print("axis=", axis)
   
    op_gather = opset.gather(data, indices, axis)
 
    Result = opset.result(op_gather, name='output_gather')
    return Model([Result], [indices], 'model_gather')
 
core = Core()
 
m = model()
compiled_model = core.compile_model(model=m, device_name="CPU")

input=np.array([[1]]).astype(np.float32)
result = compiled_model(input)[compiled_model.output(0)]

print("indices=", input)
print("---------------------->")
print("result shape=", result.shape)
print("result=", result)
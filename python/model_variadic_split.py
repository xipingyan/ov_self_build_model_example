from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import time
import os

def model():
    data = opset.parameter([6, 2], Type.i32, name = 'data')
    axis  = op.Constant(Type.i32, Shape([1]), [0])
    split_length  = op.Constant(Type.i32, Shape([3]), [3,2,1])
    print("split_length=", split_length)
   
    op_variadic_split = opset.variadic_split(data, axis, split_length)
    op_variadic_split.set_friendly_name("my_variadic_split")
 
    reshape1 = opset.reshape(op_variadic_split.output(0), op.Constant(Type.i32, Shape([1]), [6]), special_zero=False)
    reshape2 = opset.reshape(op_variadic_split.output(1), op.Constant(Type.i32, Shape([1]), [4]), special_zero=False)
    reshape3 = opset.reshape(op_variadic_split.output(2), op.Constant(Type.i32, Shape([1]), [2]), special_zero=False)

    Result1 = opset.result(reshape1, name='my_output1')
    Result2 = opset.result(reshape2, name='my_output2')
    Result3 = opset.result(reshape3, name='my_output3')
    return Model([Result1, Result2, Result3], [data], 'model_variadic_split')

def test_variadic_split():
    core = Core()
    
    m = model()

    compiled_model = core.compile_model(model=m, device_name="CPU")

    input=np.array([[1, 2],[2, 2],[3, 2],[4, 2],[5, 2],[6, 2]]).astype(np.float32)
    result = compiled_model(input)[compiled_model.output(0)]

    print("indices=", input)
    print("---------------------->")
    print("result shape=", result.shape)
    print("result=", result)

test_variadic_split()
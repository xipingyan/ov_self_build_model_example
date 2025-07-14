from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset
import numpy as np
import openvino as ov
 
def const(shape):
    w = np.random.uniform(low=-1, high=1.0, size=shape).astype(np.float32)
    return op.Constant(w)
 
def value(*shape):
    return np.random.rand(*shape).astype(np.float32)

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

# ==========================
#    input1
#      |
#   op_reshape [0,1,1]
#    /         \
# add1[0,2,1]  add2[0,1,2]
#   |           |
# op_shape1    op_shape2
#    \         /
#       Result
def model_reshape_eltwise_eltwise():
    input1 = opset.parameter([-1], Type.f32, name = 'in1')    
    bias1 = const([2, 1])

    output_shape = np.array([0, 1, 1]).astype(np.int32)
    op_reshape = opset.reshape(input1, output_shape=output_shape, special_zero=True)

    output_shape2 = np.array([2]).astype(np.int32)
    bias2 = opset.reshape(bias1, output_shape=output_shape2, special_zero=True)
   
    op_add1 = opset.add(op_reshape, bias1, auto_broadcast='numpy')
    op_add2 = opset.add(op_reshape, bias2, auto_broadcast='numpy')
 
    op_shape1 = opset.shape_of(op_add1)
    op_shape2 = opset.shape_of(op_add2)
    
    Result1 = opset.result(op_shape1, name='outp1')
    Result2 = opset.result(op_shape2, name='outp2')
    return Model([Result1, Result2], [input1], 'Model')

if __name__ == "__main__":
    print("== OV Version:", ov.get_version())
    core = Core()
    m = model_reshape_eltwise_eltwise()
    ov.save_model(m, "reshape_add_add.xml")
    input1 = value(3)

    cm_cpu = core.compile_model(model=m, device_name="CPU")
    # cm_gpu = core.compile_model(model=m, device_name="GPU", config={"INFERENCE_PRECISION_HINT":Type.f32})
    cm_gpu = core.compile_model(model=m, device_name="GPU")

    result_cpu = cm_cpu([input1])
    result_gpu = cm_gpu([input1])

    print('result_cpu:', result_cpu[0], result_cpu[1])
    print('result_gpu:', result_gpu[0], result_gpu[1])
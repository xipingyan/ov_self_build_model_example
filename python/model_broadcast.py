
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset3 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model_static():
    input = opset.parameter([3, 1, 1], Type.f32, name = 'input')

    target_shape = np.array([1, 3, 2, 2]).astype(np.int32)

    # Refer: https://docs.openvino.ai/nightly/documentation/openvino-ir-format/operation-sets/operation-specs/movement/broadcast-3.html
    # 相对于input shape，目标shape中多出来的部分，会重复原始数据。
    mode = "numpy"
    # mode = "bidirectional"
    op_broadcast = opset.broadcast(input, target_shape=target_shape, broadcast_spec=mode)
 
    Result = opset.result(op_broadcast, name='output')
    return Model([Result], [input], 'model_broadcast')

def test_broadcast():
    print("== test_broadcast ")
    core = Core()
    m = model_static()

    # Save model
    ov.save_model(m, "./tmp_model_broadcast.xml")

    compiled_model = core.compile_model(model=m, device_name="GPU")
    ireq = compiled_model.create_infer_request()

    input = np.random.rand(3, 1, 1).astype(np.float32)

    print("== input.shape=", input.shape)
    print("== input=", input)

    ov_result = ireq.infer([input])['output']
    print("---------------------->")
    print("== output.shape=", ov_result.shape)
    print("== output=", ov_result[0][0][0][:])
    print("== output=", ov_result[0][0][1][:])
    print("== output=", ov_result[0][1][0][:])

def model_dynamic():
    input = opset.parameter([-1, 1, 1], Type.f32, name = 'input')

    target_shape = np.array([1, 1, 256]).astype(np.int32)
    op_broadcast = opset.broadcast(input, target_shape=target_shape, broadcast_spec="bidirectional")
 
    Result = opset.result(op_broadcast, name='output')
    return Model([Result], [input], 'model_broadcast_dynamic')

def test_broadcast_dynamic():
    print("== test_broadcast_dynamic ")
    core = Core()
    m = model_dynamic()

    # Save model
    ov.save_model(m, "./tmp_model_broadcast_dynamic.xml")

    compiled_model = core.compile_model(model=m, device_name="GPU")
    ireq = compiled_model.create_infer_request()

    input = np.random.rand(3, 1, 1).astype(np.float32)

    print("== input.shape=", input.shape)
    print("== input=", input)

    ov_result = ireq.infer([input])['output']
    print("---------------------->")
    print("== output.shape=", ov_result.shape)
    print("== ov_result[0][0][:3]=", ov_result[0][0][:3])
    print("== ov_result[1][0][:3]=", ov_result[1][0][:3])
    print("== ov_result[2][0][:3]=", ov_result[2][0][:3])

if __name__ == "__main__":
    # test_broadcast()
    test_broadcast_dynamic()
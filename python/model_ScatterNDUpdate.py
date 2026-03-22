
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model():
    # shape=[batch, token_num]
    token_ids = opset.parameter([-1, -1], Type.i64, name = 'input')
    indices = opset.parameter([-1, 2], Type.i64, name = 'input')
    updates = opset.parameter([-1], Type.i64, name = 'input')

    scatter_nd_update_node = opset.scatter_nd_update(token_ids, indices, updates, name="scatter_nd_update")
 
    # TopK has two outputs: values (0) and indices (1)
    Result = opset.result(scatter_nd_update_node.output(0), name='output')
    return Model([Result], [token_ids, indices, updates], 'model_scatter_nd_update')
# ScatterNDUpdate
def test_scatter_nd_update():
    core = Core()
    m = model()

    # Save model
    # ov.save_model(m, "./tmp_model_scatter_nd_update.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    print("first loop")
    print("==============================================================")
    input = np.array([[1, 0, 0, 0], [5, 0, 0, 0]]).astype(np.int64)
    indices = np.array([[0, 1], [1, 1]]).astype(np.int64)
    updates = np.array([2, 3]).astype(np.int64)
    expected = np.array([[1, 2, 0, 0], [5, 3, 0, 0]]).astype(np.int64)
    # OV result
    ireq.set_input_tensor(0, ov.Tensor(input))
    ireq.set_input_tensor(1, ov.Tensor(indices))
    ireq.set_input_tensor(2, ov.Tensor(updates))
    ov_result = ireq.infer()['output']
    print("---------------------->")
    print("== Expected vs OV result:", (expected - ov_result.tolist() < 0.0001).all())

    print("first loop")
    print("==============================================================")
    input = expected
    indices = np.array([[0, 2], [1, 2]]).astype(np.int64)
    updates = np.array([3, 4]).astype(np.int64)
    expected = np.array([[1, 2, 3, 0], [5, 3, 4, 0]]).astype(np.int64)
    # OV result
    ireq.set_input_tensor(0, ov.Tensor(input))
    ireq.set_input_tensor(1, ov.Tensor(indices))
    ireq.set_input_tensor(2, ov.Tensor(updates))
    ov_result = ireq.infer()['output']
    print("---------------------->")
    print("== Expected vs OV result:", (expected - ov_result.tolist() < 0.0001).all())

if __name__ == "__main__":
    test_scatter_nd_update()
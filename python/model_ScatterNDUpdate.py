
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model():
    # shape=[batch, token_num]
    token_ids = opset8.parameter([-1, -1], Type.i64, name = 'input')
    indices = opset8.parameter([-1, 2], Type.i64, name = 'input')
    updates = opset8.parameter([-1], Type.i64, name = 'input')

    scatter_nd_update_node = opset8.scatter_nd_update(token_ids, indices, updates, name="scatter_nd_update")
 
    # TopK has two outputs: values (0) and indices (1)
    Result = opset8.result(scatter_nd_update_node.output(0), name='output')
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

# TODO
# Model input with shape [batch, token_num], and update the tensor with position id, which is more common in NLP model.
# For example, input tensor with shape [2, 4] with value [[1, 1, 1, 1], [2, 2, 2, 2]]
#   If position id [0, 0], update value [3, 4]; -> Result tensor= [[3, 1, 1, 1], [4, 2, 2, 2]]
#   If position id [2, 2], update value [8, 9]; -> Result tensor= [[1, 1, 8, 1], [2, 2, 9, 2]]
def build_model_update_tensor_with_pos_id(const_pos_id, const_update_value):
    # shape=[batch, token_num]
    token_ids = opset8.parameter([-1, -1], Type.i64, name = 'input')

    # Build indices for scatter_nd_update based on pos_id.
    token_ids_shape = opset8.shape_of(token_ids)
    batch_size = opset8.gather(token_ids_shape, new_const_1_dim(0), 0) # shape=[1], value=[batch_size]

    # Build indices as [row_idx, col_idx] for each batch.
    batch_dim_scalar = opset8.squeeze(batch_size, [0])
    start = op.Constant(Type.i64, Shape([]), [0])
    step = op.Constant(Type.i64, Shape([]), [1])
    row_idx = opset8.range(start, batch_dim_scalar, step, Type.i64)  # shape=[batch]
    row_idx_2d = opset8.unsqueeze(row_idx, 1)  # shape=[batch,1]

    col_idx = opset8.broadcast(const_pos_id, batch_size)  # shape=[batch]
    col_idx_2d = opset8.unsqueeze(col_idx, 1)  # shape=[batch,1]

    indices = opset8.concat([row_idx_2d, col_idx_2d], 1)  # shape=[batch,2]
    updates = opset8.broadcast(const_update_value, batch_size) # shape=[batch], value per batch

    scatter_nd_update_node = opset8.scatter_nd_update(token_ids, indices, updates, name="scatter_nd_update") # shape=[batch, token_num]
 
    Result = opset8.result(scatter_nd_update_node.output(0), name='output')
    return Model([Result], [token_ids], 'model_scatter_nd_update_pos_id')
def test_update_tensor_with_pos_id():
    core = Core()

    const_pos_id = op.Constant(Type.i64, Shape([3]), [1, 1, 1]) # Update token index per batch.
    const_update_value = op.Constant(Type.i64, Shape([3]), [3, 4, 5]) # Update value for batch 0 and batch 1.
    m = build_model_update_tensor_with_pos_id(const_pos_id, const_update_value)

    # Save model
    # ov.save_model(m, "./tmp_model_scatter_nd_update.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    input = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]).astype(np.int64)
    ireq.set_input_tensor(0, ov.Tensor(input))
    ov_result = ireq.infer()['output']

    print("---------------------->")
    print("== input =", input)
    expected = input.copy()
    expected[:, 1] = [3, 4, 5]
    print("== expected =", expected)
    print("---------------------->")
    print("== ov_result shape =", ov_result.shape)
    print("== ov_result value =", ov_result.tolist())
    print("== Expected vs OV result:", (expected - ov_result.tolist() < 0.0001).all())

if __name__ == "__main__":
    # test_scatter_nd_update()
    test_update_tensor_with_pos_id()
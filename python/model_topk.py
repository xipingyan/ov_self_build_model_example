
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model(FeaDim):
    # shape=[batch, sequence, fea]
    logits = opset.parameter([-1, -1, FeaDim], Type.f32, name = 'input')

    # Get axis=1's last tensor, shape=[batch, fea]
    gather = opset.gather(logits, -1, 1, name="sequence")  # shape=[batch, fea]

    topk_node = opset.topk(gather, 1, 1, 'max', 'none', name="topk")

    # TopK has two outputs: values (0) and indices (1)
    Result = opset.result(topk_node.output(1), name='output')
    return Model([Result], [logits], 'model_topk')

def test_topk():
    core = Core()
    FeaDim = 4
    m = model(FeaDim=FeaDim)

    # Save model
    # ov.save_model(m, "./tmp_model_topk.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    # input.shape = [2, 1, FeaDim]
    input = np.array([[[1, 2, 3, 4], [21, 88, 23, 24]], [[5, 6, 8, 7], [99, 26, 28, 27]]]).astype(np.float32)
    assert(FeaDim == input.shape[-1])
    
    print("== input.shape =", input.shape)
    expected_max_value = input[:, -1, :].max(axis=1, keepdims=True)
    expected_max_value_indices = input[:, -1, :].argmax(axis=1, keepdims=True)
    print("== expected_max_value =\n", expected_max_value)
    print("== expected_max_value_indices =\n", expected_max_value_indices)

    # OV result
    ov_result = ireq.infer(input)['output']
    print("---------------------->")
    print("== ov_result shape =", ov_result.shape)
    print("== ov_result =", ov_result)

    print("== Expected vs OV result:", (expected_max_value_indices - ov_result.tolist() < 0.0001).all())

if __name__ == "__main__":
    test_topk()
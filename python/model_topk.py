
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model(FeaDim):
    # shape=[batch, fea]
    logits = opset.parameter([-1, FeaDim], Type.f32, name = 'input')

    # TopK on axis 0: topk=1, axis=0, mode='max', sort='none'
    topk = 1
    axis = 1
    mode = 'max'
    sort = 'none'
    name = "topk"
    topk_node = opset.topk(logits, topk, axis, mode, sort, name=name)
 
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

    # input.shape = [2, FeaDim]
    input = np.array([[1, 2, 3, 4], [5, 6, 8, 7]]).astype(np.float32)
    assert(FeaDim == input.shape[-1])
    
    print("== input.shape=", input.shape)
    # Expected result, max value's indices on axis 1: [[3], [2]]
    # input.max(axis=1, keepdims=True) = [[4], [8]]
    # input.argmax(axis=1, keepdims=True) = [[3], [2]]   // Get the indices.
    expected_max_value = input.max(axis=1, keepdims=True)
    expected_max_value_indices = input.argmax(axis=1, keepdims=True)
    print("== Expected result = input.max(axis=1, keepdims=True) =\n", expected_max_value)
    print("== Expected result = input.argmax(axis=1, keepdims=True) =\n", expected_max_value_indices)

    # OV result
    ov_result = ireq.infer(input)['output']
    print("---------------------->")
    print("== OV result =\n", ov_result)

    print("== Expected vs OV result:", (expected_max_value_indices - ov_result.tolist() < 0.0001).all())

if __name__ == "__main__":
    test_topk()
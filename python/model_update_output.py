
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

# Src Model
# =============================
#     input[-1,-1,FeaDim]
#            |
#           Add(bias=const)
#            |
#          Output
def model(FeaDim, bias):
    # shape=[batch, seq, fea]
    input_ids = opset.parameter([-1, -1, FeaDim], Type.f32, name = 'input')    
    add = opset.add(input_ids, op.Constant(Type.f32, Shape([1]), [bias]))
    Result = opset.result(add, name='output')
    return Model([Result], [input_ids], 'model_add')

# Get last seq fea, Function=org_output[:,-1,:], for example:
# Orginal output shape=[-1,-1,FeaDim]
# New output shpae=[-1,FeaDim]

# Updated Model Graph
# =============================
#     input[-1,-1,FeaDim]
#            |
#         Add(bias=const)
#          /            \
#      ShapeOf(3Dim)     \
#         |               \
#      Gather(Get middle)  |
#         |               /
#        Add(-1)         /
#            \          /
#            Gather(axis=1)[-1,1,FeaDim]
#                 |
#             Reshape[-1, FeaDim]
#                 |
#              Output
def update_model_get_logits(ov_model:ov.Model, FeaDim):    
    org_output = ov_model.get_result()
    org_output_input = org_output.input(0).get_source_output()

    # Remove old output node.
    ov_model.remove_result(org_output)

    # Take old output node's input as new input.
    input_ids_shape = opset.shape_of(org_output_input)

    # Get middle value from [-1,-1,FeaDim], for example: get 2 from shape[1,2,3]
    last_seq_id = opset.gather(input_ids_shape, new_const_1_dim(1), new_const_1_dim(0))
    # idx start from 0, so need to reduce 1.
    indics = opset.add(last_seq_id, op.Constant(Type.i64, Shape([1]), [-1]))

    # Get last seq fea, same to input[:,-1,:]
    last_logits = opset.gather(org_output_input, indics, new_const_1_dim(1))

    # Reshape from [-1, 1, FeaDim] to [-1, FeaDim]
    reshape = opset.reshape(last_logits, opset.constant([-1, FeaDim], dtype=ov.Type.i64), special_zero=True)

    result_new = opset.result(reshape, name='output_new')

    return ov.Model([result_new], [n.get_node() for n in ov_model.inputs], "model_updated")

def unit_test():
    core = Core()
    FeaDim = 4
    bias = -0.1
    m_org = model(FeaDim=FeaDim, bias=bias)

    ov.save_model(m_org, "./tmp_model_org.xml")
    new_model = update_model_get_logits(m_org, FeaDim=FeaDim)
    ov.save_model(new_model, "./tmp_model_new.xml")

    compiled_model = core.compile_model(model=new_model, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    # input.shape= (2, 3, 4)
    input = np.array([[[111, 112, 113, 4], [121, 122, 123, 4], [131, 132, 133, 4]], 
                      [[211, 212, 213, 4], [221, 222, 223, 4], [231, 232, 233, 4]]]).astype(np.float32)
    assert(FeaDim == input.shape[-1])
    
    print("== input.shape=", input.shape)
    print("== input[:,-1,:].shape=", input[:,-1, :].shape)
    expected_result = input[:,-1, :] + bias
    print("== Expected result = input[:,-1,:] =\n", expected_result)

    # ov_result = compiled_model(input)[compiled_model.output(0)] # or bellow.
    ov_result = ireq.infer(input)['output_new']
    print("---------------------->")
    print("== result.shape=", ov_result.shape)
    print("== OV result =\n", ov_result)

    print("== Expected vs OV result:", (expected_result-ov_result.tolist() < 0.0001).all())

if __name__ == "__main__":
    unit_test()
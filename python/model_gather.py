
import numpy as np

import openvino as ov
from openvino import Core, Model, Type, Shape, op
from openvino import opset8 as opset

def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def model(FeaDim):
    # shape=[batch, seq, fea]
    input_ids = opset.parameter([-1, -1, FeaDim], Type.i32, name = 'input')
    input_ids_shape = opset.shape_of(input_ids)

    # Get middle value from [-1,-1,FeaDim], for example: get 2 from shape[1,2,3]
    last_seq_id = opset.gather(input_ids_shape, new_const_1_dim(1), new_const_1_dim(0))

    # idx start from 0, so need to reduce 1.
    indics = opset.add(last_seq_id, op.Constant(Type.i64, Shape([1]), [-1]))

    # Get last seq fea, same to input_ids[:,-1,:]
    last_logits = opset.gather(input_ids, indics, new_const_1_dim(1))
    
    if 0: # Infer to get reshaped target shape.
        fea_dim = opset.gather(input_ids_shape, new_const_1_dim(2), new_const_1_dim(0))
        fea_dim= opset.add(fea_dim, op.Constant(Type.i64, Shape([1]), [-1]))
        neg_one = opset.constant([-1], dtype=Type.i64)
        target_shape = opset.concat([neg_one, fea_dim], axis=0)
    else: # Get it from const.
        target_shape = opset.constant([-1, FeaDim], dtype=ov.Type.i64)

    # Reshape from [-1, 1, FeaDim] to [-1, FeaDim]
    reshape = opset.reshape(last_logits, target_shape, special_zero=True)
 
    Result = opset.result(reshape, name='output')
    return Model([Result], [input_ids], 'model_gather')

def test_gather():
    core = Core()
    FeaDim = 4
    m = model(FeaDim=FeaDim)

    # Save model
    # ov.save_model(m, "./tmp_model_gather.xml")

    compiled_model = core.compile_model(model=m, device_name="CPU")
    ireq = compiled_model.create_infer_request()

    # input.shape= (2, 3, 4)
    input = np.array([[[111, 112, 113, 4], [121, 122, 123, 4], [131, 132, 133, 4]], 
                      [[211, 212, 213, 4], [221, 222, 223, 4], [231, 232, 233, 4]]]).astype(np.int32)
    assert(FeaDim == input.shape[-1])
    
    print("== input.shape=", input.shape)
    print("== input[:,-1,:].shape=", input[:,-1, :].shape)
    expected_result = input[:,-1, :]
    print("== Expected result = input[:,-1,:] =\n", expected_result)

    # result = compiled_model(input)[compiled_model.output(0)] # or bellow.
    ov_result = ireq.infer(input)['output']
    print("---------------------->")
    print("== result.shape=", ov_result.shape)
    print("== OV result =\n", ov_result)

    print("== Expected vs OV result:", (expected_result-ov_result.tolist() < 0.0001).all())

if __name__ == "__main__":
    test_gather()
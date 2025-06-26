
import openvino as ov
import numpy as np
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op

def my_model():
    input = opset.parameter([1, 256, 32, 32], Type.f32, name='input')

    weight_arr = np.random.uniform(low=-1, high=1.0, size=[1024,256,1,1]).astype(np.float32)
    weight = opset.constant(weight_arr, Type.f32, name='weight')

    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    dilations = [1, 1]
    conv = opset.convolution(input, weight, strides, pads_begin, pads_end, dilations)

    add = opset.add(conv, np.random.uniform(low=-1, high=1.0, size=[1,1024,1,1]).astype(np.float32), name='op_add')

    op_gelu = opset.gelu(add, approximation_mode="ERF")
 
    Result = opset.result(op_gelu, name='output')
    return Model([Result], [input], 'model_gelu')

def add_new_output(ov_model:ov.Model, name):
    found_node_output = None
    for op in ov_model.get_ordered_ops():
        if op.get_friendly_name() == name:
            # Assuming op has one output, take its first output port
            found_node_output = op.output(0)
            break

    new_output_name = name + "_output"
    if found_node_output:
        # Create a new Result node connected to the found output
        new_result = ov.opset12.result(found_node_output)
        new_result.set_friendly_name(new_output_name) # Set a friendly name
        new_result.output(0).set_names({new_output_name}) # Set tensor names


        # Add the new result node to the model's outputs
        ov_model.add_results([new_result])
        print(f"== Added new output: {new_output_name}")
    else:
        print(f"== Error: Could not find node '{name}' to add as a new output.")

    return ov_model

def test():
    model = my_model()

    new_result_name="op_add"
    model = add_new_output(model, new_result_name)

    cm = ov.compile_model(model, "GPU")

    import os
    if os.path.exists("tmp_input.npy"):
        input = np.load("tmp_input.npy")
    else:
        input = np.random.uniform(low=0, high=1.0, size=(1, 256, 32, 32)).astype(np.float32)
        np.save("tmp_input.npy", input)

    output = cm(input)
    outp_0=output[0]
    outp_1=output[new_result_name+"_output"]

    print("outp_0.shape=", outp_0.shape)
    print("outp_1.shape=", outp_1.shape)

if __name__ ==  "__main__":
    test()
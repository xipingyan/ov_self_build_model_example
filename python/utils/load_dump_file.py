import numpy as np

# load openvino GPU dump's input/output data.
def load_dump_file(fn:str, new_shape=None, shape_dim=1):
    # Read file
    with open(fn, "r") as file:
        first_line = file.readline()
        # parse shape: shape: [b:1, f:256, x:32, y:32, z:1, w:1, u:1, v:1, g:1]
        first_line = first_line.split(']')[0].split('[')[1]
        first_line = first_line.split(',')
        shape = []
        for itm in first_line:
            shape.append(int(itm.split(':')[1]))
        print(" == shape =", shape)

        dt_list = file.readlines()
        converted_list = [float(item.strip()) for item in dt_list]
        arr = np.array(converted_list, dtype=np.float32)
        if new_shape != None:
            arr = arr.reshape(new_shape)
        elif shape_dim < len(shape):
            arr = arr.reshape(shape[:shape_dim])
        return arr

if __name__ == "__main__":
    input_root = "./test_data/program1_network1_0_convolution__backbone_stages.3_op_list.2_main_inverted_conv_conv_Conv_WithoutBiases"
    
    inp_1 = load_dump_file(input_root + "_src0.txt", shape_dim=4)
    print(inp_1.shape)

import openvino as ov
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import os
import sys
sys.path.append("../../")
from utils.comm_pt import cache_randn_1d, cache_randn_3d

from openvino import opset8 as opset
from openvino import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize

def run_ov_model(input_pt, onnx_model_fn):
    print("== Start to run OV model.")
    ext_path = "./cpu/build/libopenvino_custom_add_extension.so"
    # ext_path = os.getenv('CUSTOM_OP_LIB')

    # OV model:
    core = Core()
    core.add_extension(ext_path)

    model = core.read_model(onnx_model_fn)

    if not os.path.exists("export_ov_model"):
        os.mkdir("export_ov_model")
    print("== Start to save OpenVINO IR.")
    ov.save_model(model, "export_ov_model/custom_model.xml")

    compiled_model = core.compile_model(model, 'CPU')

    input = np.array(input_pt.cpu().tolist()).astype(np.float32)
    out = compiled_model({"input": input})
    return out["output"].tolist()

class MyAddPyOP(torch.autograd.Function):
    # Overriding the symbolic method in 'torch.autograd' Function is crucial 
    # when you want to export your custom PyTorch operation to formats like ONNX 
    @staticmethod
    def symbolic(g, x, bias):
        bias = torch.tensor(bias)
        bias = g.op("Constant", value_t=bias)
        # g.op(op_name, input1...)
        return g.op('MyAdd', x, bias)

    @staticmethod
    def forward(self, x, bias):
        y = x + bias
        return y
    
class MyPytorchModel(nn.Module):
    def __init__(self, last_bias):
        super(MyPytorchModel, self).__init__()
        self.last_bias = last_bias
        self.norm = F.layer_norm
        self.normalized_shape_a = (10,) # Must be const.
        self.default_eps = 1e-5
        self.weight_a = cache_randn_1d([20, 5, 10], "./tmp/weight.pt")
        self.bias_a = cache_randn_1d([20, 5, 10], "./tmp/bias.pt")
        self.my_add_py_op = MyAddPyOP

    def forward(self, x):
        r1 = self.norm(x.contiguous(), self.normalized_shape_a, self.weight_a, self.bias_a, self.default_eps)
        r2 = self.my_add_py_op.apply(r1, self.last_bias)
        return r2

if __name__ == "__main__":
    # Original model(pytorch model)
    model_pt = MyPytorchModel(0.1)
    
    batch, sentence_length, embedding_dim = 20, 5, 10
    inp = cache_randn_3d(batch, sentence_length, embedding_dim, "./tmp/input_3d.pt")
    model_pt.eval()

    # Export ONNX
    onnx_model_fn = 'my_model_static.onnx'
    
    input_names = ["input"]
    output_names = ["output"]

    # if False: # With static shape.
    if True: # With dynamic shape
        onnx_model_fn = 'my_model_dynamic.onnx'
        dynamic_axes = {
            "input": {0: "batch"},  # 0 is the index of the batch dimension
            "output": {0: "batch"} # 0 is the index of the batch dimension
        }
        with torch.no_grad():
            torch.onnx.export(model_pt, inp, onnx_model_fn,
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes,
                            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
    else:
        with torch.no_grad():
            torch.onnx.export(model_pt, inp, onnx_model_fn,
                            input_names=input_names,
                            output_names=output_names,
                            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)

    ref = model_pt(inp)
    rslt_ov = torch.tensor(run_ov_model(inp, onnx_model_fn))
    # print("== ref=", ref[0][0][:3])
    # print("== rslt_ov=", rslt_ov[0][0][:3])
    print("== Compare result, Torch VS OV =", 
          torch.isclose(ref, rslt_ov, 1e-5, 1e-5).all().numpy())
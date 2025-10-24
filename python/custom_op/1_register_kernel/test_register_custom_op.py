# Dependencies:
# pip install torch onnx
import openvino as ov
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import copy

import os
import sys
sys.path.append("../../")
from utils.comm_pt import cache_randn_1d, cache_randn_3d

from openvino import opset8 as opset
from openvino import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize

TMP_DIR='./tmp/model_custom_op'
os.makedirs(TMP_DIR, exist_ok=True)

def run_ov_model(inputs_pt:list, onnx_model_fn, device='CPU', dynamic_shape=False):
    print(" == Start to run OV model.")
    ext_path = "./cpu/build/libopenvino_custom_add_extension.so"
    # ext_path = os.getenv('CUSTOM_OP_LIB')

    # OV model:
    core = Core()
    core.add_extension(ext_path)
    
    print(f" == device = '{device}'.")
    if device == 'GPU':
        core.set_property("GPU", {"CONFIG_FILE": "./gpu/custom_add.xml"})

    ov_ir = f"{TMP_DIR}/export_ov_model/openvino_model_{device}_{dynamic_shape}.xml"
    if onnx_model_fn is None:
        print(f"  == Start to load {ov_ir}")
        model = core.read_model(ov_ir)
    else:
        print(f"  == Start to load {onnx_model_fn}")
        model = core.read_model(onnx_model_fn)

        if not os.path.exists("export_ov_model"):
            os.mkdir("export_ov_model")
        print(f"  == Start to save OpenVINO IR: {ov_ir}")
        ov.save_model(model, ov_ir)

    compiled_model = core.compile_model(model, device)

    if device == 'GPU':
        print(f" == Release model for gpu plugin to reduce cpu host mem.")
        model = None
        print(f" == Release model done.")

    outputs = []
    for id, input_pt in enumerate(inputs_pt):
        input = np.array(input_pt.cpu().tolist()).astype(np.float32)
        
        print(f"** infer {id} ")
        out = compiled_model({"input": input})
        outputs.append(copy.deepcopy(out["output"].tolist()))
    return outputs

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
        # print("bias=", bias, y[0][0][0:3])
        return y
    
class MyPytorchModel(nn.Module):
    def __init__(self, last_bias):
        super(MyPytorchModel, self).__init__()
        self.last_bias = last_bias
        self.norm = F.layer_norm
        self.normalized_shape_a = (10,) # Must be const.
        self.default_eps = 1e-5
        self.weight_a = cache_randn_1d([10], f"{TMP_DIR}/weight.pt")
        self.bias_a = cache_randn_1d([10], f"{TMP_DIR}/bias.pt")
        self.my_add_py_op = MyAddPyOP

    def forward(self, x):
        print(f" == MyPytorchModel::forward, input shape: {x.shape}, x[:3]={x[0][0][:3]}")
        r1 = self.norm(x.contiguous(), self.normalized_shape_a, self.weight_a, self.bias_a, self.default_eps)
        r2 = self.my_add_py_op.apply(r1, self.last_bias)
        return r2

def main(device, dynamic_shape):
    print(f"== Start to test custom op with device='{device}', dynamic_shape='{dynamic_shape}'")

    # Original model(pytorch model)
    model_pt = MyPytorchModel(0.1)

    batch, sentence_length, embedding_dim = 20, 5, 10

    if not dynamic_shape:
        inputs = [cache_randn_3d(batch, sentence_length, embedding_dim, f"{TMP_DIR}/input_static_3d_0.pt", dtype=torch.float32),
                  cache_randn_3d(batch, sentence_length, embedding_dim, f"{TMP_DIR}/input_static_3d_1.pt", dtype=torch.float32)]
    else:
        inputs = [cache_randn_3d(batch, sentence_length, embedding_dim, f"{TMP_DIR}/input_dyn_3d_0.pt", dtype=torch.float32),
                  cache_randn_3d(10, sentence_length, embedding_dim,
                                 f"{TMP_DIR}/input_dyn_3d_1.pt", dtype=torch.float32),
                  cache_randn_3d(10, sentence_length, embedding_dim,
                                 f"{TMP_DIR}/input_dyn_3d_1_1.pt", dtype=torch.float32),
                  cache_randn_3d(2, sentence_length, embedding_dim,
                                 f"{TMP_DIR}/input_dyn_3d_2.pt", dtype=torch.float32)
                  ]

    model_pt.eval()
    
    onnx_model_fn = f'my_model_{device}_{dynamic_shape}.onnx'
    
    DirectCallOV = True
    DirectCallOV = False
    if not DirectCallOV:
        # Export ONNX
        input_names = ["input"]
        output_names = ["output"]

        if dynamic_shape == 'static':
            with torch.no_grad():
                torch.onnx.export(model_pt, inputs[0], onnx_model_fn,
                                input_names=input_names,
                                output_names=output_names,
                                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
        else:
            dynamic_axes = {
                "input": {0: "batch"}, # 0 is the index of the batch dimension
                "output": {0: "batch"} # 0 is the index of the batch dimension
            }
            with torch.no_grad():
                torch.onnx.export(model_pt, inputs[0], onnx_model_fn,
                                input_names=input_names,
                                output_names=output_names,
                                dynamic_axes=dynamic_axes,
                                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)

    print(f"== Start to run torch model.")
    outp_refs = []
    for inp in inputs:
        ref = copy.deepcopy(model_pt(inp))
        outp_refs.append(ref)

    outps_ov = run_ov_model(inputs, None if DirectCallOV else onnx_model_fn, device, dynamic_shape)
    outps_ov = [torch.tensor(rslt_ov) for rslt_ov in outps_ov]

    for idx, inp in enumerate(inputs):
        print(f"  == input[{idx}] =", inp[0][0][:3])

    for idx, outp in enumerate(outp_refs):
        print(f"  == output_ref[{idx}] =", outp[0][0][:3])
   
    for idx, outp in enumerate(outps_ov):
        print(f"  == output_ov[{idx}] =", outp[0][0][:3])

    rtol,atol=(1e-3,1e-3) if device == 'GPU' else (1e-5,1e-5)
    if device == 'GPU':
        print("== Warning: Intel GPU default accuracy is fp16, so the accuracy threshold is 1e-3.")
    
    assert(len(outp_refs)==len(outps_ov))
    for idx in range(len(outps_ov)):
        print(f"== Compare result with T: {rtol, atol}, Torch VS OV =", 
              torch.isclose(outp_refs[idx], outps_ov[idx], rtol, atol).all().numpy())

if __name__ == "__main__":
    print("ov version: ", ov.get_version())
    print("pid: ", os.getpid())

    devices_list = ["CPU", "GPU"]
    dynamic_list = [False, True]
    devices_list = ["GPU"]
    devices_list = ["CPU"]
    dynamic_list = [True]
    for dev in devices_list:
        for dynamic in dynamic_list:
            print(f"**** main dev={dev}, dynamic_shape={dynamic} ****")
            main(device=dev, dynamic_shape=dynamic)

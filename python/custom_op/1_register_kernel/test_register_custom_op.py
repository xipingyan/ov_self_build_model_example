import openvino as ov
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
sys.path.append("../../")
from utils.comm_pt import cache_randn_1d, cache_randn_3d

from openvino import opset8 as opset
from openvino import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize

def ov_model(weight_a:torch.tensor, bias_a, default_eps, normalized_shape_a):
    input = opset.parameter([-1, -1, 10], Type.f32, name='input0')

    axes_node = opset.constant([2], dtype=ov.Type.i64, name="mvn_axes")
    normalize_variance = True
    mvn_node = opset.mvn(input, axes_node, normalize_variance, default_eps, eps_mode='inside_sqrt')

    const_weight = opset.constant(weight_a.cpu().tolist(), Type.f32, "weight")
    multiple_node = opset.multiply(mvn_node, const_weight)

    const_bias = opset.constant(bias_a.cpu().tolist(), Type.f32, "weight")
    add_node = opset.add(multiple_node, const_bias)

    Result = opset.result(add_node, name='output')
    return Model([Result], [input], 'model_layer_norm')

def test_pt_ov_model():
    # Torch model
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = cache_randn_3d(batch, sentence_length, embedding_dim, "./tmp/input_3d.pt")
    default_eps = 1e-5
    normalized_shape_a = (embedding.shape[-1],)
    weight_a = cache_randn_1d(normalized_shape_a, "./tmp/weight.pt")
    bias_a = cache_randn_1d(normalized_shape_a, "./tmp/bias.pt")
    result_pt = F.layer_norm(embedding, normalized_shape_a, weight=weight_a, bias=bias_a, eps=default_eps)

    # Custom OP of Pytorch (Add const)
    # ======================================
    const_bias = 0.1
    result_pt = result_pt + const_bias
    # ======================================
    print(f"== **************************************")
    print(f"== result_pt shape: {result_pt.shape}")
    print(f"== first line result:\n{result_pt.cpu().tolist()[0][0][:6]}")
    print(f"== last line result:\n{result_pt.cpu().tolist()[0][4][:6]}")

    # OV model:
    core = Core()
    m = ov_model(weight_a, bias_a, default_eps, normalized_shape_a)
    # Register custom OP of OV

    compiled_model = core.compile_model(model=m, device_name="CPU")

    input = np.array(embedding.cpu().tolist()).astype(np.float32)
    result_ov = compiled_model(input)
    print(f"== **************************************")
    print(f"== result_ov shape: {result_ov['output'].shape}")
    print(f"== first line result:\n{result_ov['output'].tolist()[0][0][:6]}")
    print(f"== last line result:\n{result_ov['output'].tolist()[0][4][:6]}\n")

    output_node = result_ov['output']
    ov_pt_output = torch.tensor(output_node.tolist(), dtype=torch.float32)
    # ov_pt_output = ov_pt_output + const_bias
    bclose = torch.isclose(ov_pt_output, result_pt, 1e-5, 1e-5).all()
    print(f"== **************************************")
    print(f"== compare 2 result, torch.isclose(ov_pt_output, result_pt, 1e-5, 1e-5).all() = {bclose}")
    print(f"== Test {'pass' if bclose else 'fail'}.")

if __name__ == "__main__":
    test_pt_ov_model()
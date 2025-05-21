# Torch layer_norm VS OpenVINO MVN
import torch
import torch.nn as nn
import torch.nn.functional as F

from openvino import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino as ov
import openvino.properties.hint as hints
# from openvino.runtime.op import util as op_util
from openvino import opset8 as opset
# from openvino.runtime.passes import Manager
import numpy as np
import time
import os

# Torch: layer_norm -> OV: mvn+multply+add
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

def cache_randn(batch, sentence_length, embedding_dim):
    dump_fn = "embedding.pt"
    if os.path.exists(dump_fn):
        embedding = torch.load(dump_fn)
    else:
        embedding = torch.randn(batch, sentence_length, embedding_dim, dtype=torch.float32)
        torch.save(embedding, dump_fn)
    return embedding

def cache_randn_1d(embedding_dim, cache_fn):
    dump_fn = cache_fn
    if os.path.exists(dump_fn):
        embedding = torch.load(dump_fn)
    else:
        embedding = torch.randn(embedding_dim, dtype=torch.float32)
        torch.save(embedding, dump_fn)
    return embedding

def test_layer_norm():
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = cache_randn(batch, sentence_length, embedding_dim)

    default_eps = 1e-5
    normalized_shape_a = (embedding.shape[-1],)
    weight_a = cache_randn_1d(normalized_shape_a, "weight.pt")
    bias_a = cache_randn_1d(normalized_shape_a, "bias.pt")
    result_pt = F.layer_norm(embedding, normalized_shape_a, weight=weight_a, bias=bias_a, eps=default_eps)
    print(f"== result_pt shape: {result_pt.shape}")
    print(f"== first line result:\n{result_pt.cpu().tolist()[0][0][:6]}")
    print(f"== last line result:\n{result_pt.cpu().tolist()[0][4][:6]}")

    core = Core()
    m = ov_model(weight_a, bias_a, default_eps, normalized_shape_a)
    compiled_model = core.compile_model(model=m, device_name="CPU")

    input = np.array(embedding.cpu().tolist()).astype(np.float32)
    result_ov = compiled_model(input)
    output_node = result_ov['output']
    ov_pt_output = torch.tensor(output_node.tolist(), dtype=torch.float32)
    print(f"== result_ov shape: {result_ov['output'].shape}")
    print(f"== first line result:\n{result_ov['output'].tolist()[0][0][:6]}")
    print(f"== last line result:\n{result_ov['output'].tolist()[0][4][:6]}\n")

    bclose = torch.isclose(ov_pt_output, result_pt, 1e-5, 1e-5).all()
    print(f"== compare 2 result, torch.isclose(ov_pt_output, result_pt, 1e-5, 1e-5).all() = {bclose}")

if __name__ == "__main__":
    print("== Start to test: 'torch.nn.LayerNorm' VS 'OV implementation(MVN+Multply+Add)'")
    test_layer_norm()
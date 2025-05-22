import torch
import os
import numpy as np

def mk_root_dir(root):
    if not os.path.exists(root) and root != '':
        os.mkdir(root)

# llm: single head case.
def cache_randn_3d(batch, sentence_length, embedding_dim, cache_fn:str, dtype=torch.float32, regenerate=False):
    root, fn = os.path.split(cache_fn)
    mk_root_dir(root)

    if os.path.exists(cache_fn) and regenerate is False:
        tensor_3d = torch.load(cache_fn)
    else:
        tensor_3d = torch.randn(batch, sentence_length, embedding_dim, dtype=torch.float32)
        torch.save(tensor_3d, cache_fn)
    return tensor_3d

def cache_randn_1d(fea_dim, cache_fn, dtype=torch.float32, regenerate=False):
    root, fn = os.path.split(cache_fn)
    mk_root_dir(root)

    if os.path.exists(cache_fn) and regenerate is False:
        tensor_1d = torch.load(cache_fn)
    else:
        tensor_1d = torch.randn(fea_dim, dtype=torch.float32)
        torch.save(tensor_1d, cache_fn)
    return tensor_1d

if __name__ == "__main__":
    # inp = cache_randn_3d(1,2,3, "tmp/input.pt")
    # inp = cache_randn_1d(3, "input_1d.pt")
    inp = cache_randn_1d(3, "./input_1d.pt")
    print(f"inp type: {type(inp)}, {inp.shape}, {inp.dtype}")
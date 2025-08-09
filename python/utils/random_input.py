import numpy as np
from openvino import Tensor
import os

def generate_random_array(*args, type=np.float32, cache_prefix='tmp', enable_cache=True, remove_cache=False):
    data = (np.random.randn(*args)*255).astype(type)
    if enable_cache and cache_prefix != None:
        cache_name = cache_prefix + ".npy"
        if remove_cache:
            os.remove(cache_name)
        if os.path.exists(cache_name):
            print(f"    Use cached data: {data.shape}, type={type}")
            return np.load(cache_name)
        else:
            np.save(cache_name, data)
    print(f"    Generate new data: {data.shape}, type={type}")
    return data

def array_to_tensor(arr):
    return Tensor(arr)

if __name__ == "__main__":
    input = generate_random_array(2, 3, 4, type=np.uint8)
    print(input.shape, input)

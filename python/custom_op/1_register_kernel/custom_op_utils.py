import openvino as ov
import numpy as np
import torch
import torch.nn as nn

# Custom op, can't export to ov directly. Got error.
def export_torch_model_to_ov(model_pt:nn.Module, dummy_input, OUTPUT_PATH_XML:str):
    model_pt.eval()
    ov_model = ov.convert_model(model_pt, example_input=dummy_input)
    ov.save_model(
        model=ov_model, 
        output_model=OUTPUT_PATH_XML,
        # Optional: Set to False to keep weights in FP32
        compress_to_fp16=True 
    )
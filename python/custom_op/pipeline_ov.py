import openvino as ov
import os
import numpy as np

def run_pipeline(model_fn):
    core = ov.Core()
    model = core.read_model(model_fn)
    cm = ov.compile_model(model, 'GPU')

    input=np.random.rand(1,3,384,640).astype(np.float32)
    output = cm(input)

    print(f"output['output'] = {output['output'].shape}")
    print(f"output['481'] = {output['481'].shape}")
    print(f"output['490'] = {output['490'].shape}")
    print(f"output['499'] = {output['499'].shape}")

if __name__ == "__main__":
    model_fn = 'yolov7-pose_without_YoloPoseLayer_TRT.onnx'
    model_fn = "./output_ir/yolov7_ov.xml"
    run_pipeline(model_fn)
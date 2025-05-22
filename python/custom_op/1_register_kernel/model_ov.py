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

def run_pipeline_with_custom_op(model_fn):
    core = ov.Core()
    core.set_property("GPU", {"CONFIG_FILE": "./gpu_custom_op/custom_op.xml"})
    model = core.read_model(model_fn)
    cm = ov.compile_model(model, 'GPU')

    input=np.random.rand(1,3,384,640).astype(np.float32)
    output = cm(input)
    print(f"output['output'] = {output['output'].shape}")
    
    # <Tensor arg-index="0" type="output/sink_port_0" port-index="0" format="BFYX"/>
    # <Tensor arg-index="1" type="481/sink_port_0" port-index="0" format="BFYX"/>
    # <Tensor arg-index="2" type="490/sink_port_0" port-index="0" format="BFYX"/>
    # <Tensor arg-index="3" type="499/sink_port_0" port-index="0" format="BFYX"/>

if __name__ == "__main__":
    model_fn = 'yolov7-pose_without_YoloPoseLayer_TRT.onnx'
    model_fn = "./output_ir/yolov7_ov.xml"
    run_pipeline(model_fn)
import openvino as ov
import os

def cvt_to_ov(model_fn):
    out_path = "./output_ir"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    ov_model = ov.convert_model(model_fn)
    ov.save_model(ov_model, out_path+"/yolov7_ov.xml")

if __name__ == "__main__":
    model_fn = 'yolov7-pose_without_YoloPoseLayer_TRT.onnx'
    cvt_to_ov(model_fn)
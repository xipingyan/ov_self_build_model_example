model="./hema-bhf-joint-s-t20231220-2-640x384-dynamic_1.onnx"
# model="./yolov7-pose_without_YoloPoseLayer_TRT.onnx"

# benchmark_app -d GPU -m $model -data_shape images[1,3,384,640] -nireq 1 -nstreams 1 -nthreads 1 -hint none -niter 1

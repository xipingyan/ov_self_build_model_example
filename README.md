# ov_slef_build_model_example
This is quick verification example for OpenVINO specific layer or node. We can construct a model and inference this model.

# How to build && run test
Build script. Please install your OpenVINO firstly.
```
mkdir build && cd build
source ../../openvino/build/install/setupvars.sh # Your OV Env.
cmake ..
make -j8
./testapp
```
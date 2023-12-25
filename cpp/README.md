# How to test

Please install your OpenVINO firstly.
```
mkdir build && cd build
source ../../../openvino/build/install/setupvars.sh # Your OV Env.
cmake ..
make -j8
./testapp
```
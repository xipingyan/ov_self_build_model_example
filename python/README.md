# How to test

Please install your OpenVINO firstly.
```
python3 -m venv python-env
source python-env/bin/activate
source ../../openvino/build/install/setupvars.sh # Your OV Env.
pip install numpy

<!-- Test -->
python model_conv_bias_sum_reshape.py
```
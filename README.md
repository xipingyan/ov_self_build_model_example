# ov_slef_build_model_example
This is quick verification example for OpenVINO specific layer or node. We can construct a model and inference this model.

# CPP
Test model constructed via cpp interface.

# Python
Test model constructed via python interface.

    cd ov_self_build_model_example/python
    python3 -m venv python-env
    source python-env/bin/activate

    <!-- update openvino -->
    pip install openvino
    python -c "import openvino as ov; print(ov.get_version())"
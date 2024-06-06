import numpy as np
import openvino as ov

root_path='path'
# read model
model = ov.Core().read_model(f"{root_path}/openivno_model.xml")

# create inputs
inputs = {}
for val in model.inputs:
    cur_fn = f"{root_path}/inputs/{val.any_name.replace('/', '_')}.npy"
    print("cur_fn=", cur_fn)
    value = np.load(cur_fn)
    inputs[val.any_name] = value

# collect output names
output_names = set()
for output in model.outputs:
    for name in output.get_names():
        if name.startswith("Result_"):
            output_names.add(name)

compiled_model = ov.compile_model(model, device_name="CPU")
model_outputs = compiled_model(inputs)

assert len(model_outputs) == len(output_names)

model_output_names = set()
for tensor, value in model_outputs.items():
    for tensor_name in tensor.get_names():
        model_output_names.add(tensor_name)

missing_names = output_names - model_output_names

print("Output names:")
for name in output_names:
    print(name)
print("\nModel output names:")
for name in model_output_names:
    print(name)
print("\nMissing names:")
for name in missing_names:
    print(name)

assert len(missing_names) == 0

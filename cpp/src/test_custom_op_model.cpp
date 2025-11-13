#include "private.hpp"
#include "utils.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>

static ov::TensorVector test_model_device(std::string device, std::string model_xml, std::vector<ov::Tensor> inputs, std::string &extension_path, std::string &gpu_cfg)
{
    std::cout << "== Start test: =========== " << __FUNCTION__ << " ============" << std::endl;
    ov::Core core = ov::Core();
    core.add_extension(extension_path);
    if (device == "GPU") {
        ov::AnyMap properties;
        properties["CONFIG_FILE"] = gpu_cfg;
        core.set_property("GPU", properties);
    }
    auto model = core.read_model(model_xml);

    // Compile model.
    std::cout << "  == compile_model with device: " << device << std::endl;
    auto compiledModel = core.compile_model(model, device);
    std::cout << "  == create_infer_request" << std::endl;
    auto inferRequest = compiledModel.create_infer_request();

    for (size_t i = 0; i < inputs.size(); i++) {
        std::cout << "  == set_input_tensors" << std::endl;
        inferRequest.set_input_tensors(i, {inputs[i]});
    }

    // Inference
    std::cout << "  == Start infer. device=" << device << std::endl;
    inferRequest.infer();

    ov::TensorVector outputs;
    for (size_t i = 0; i < model->get_output_size(); i++) {
        auto outTensor = inferRequest.get_output_tensor(i);
        outputs.push_back(std::move(outTensor));
    }
    return outputs;
}

static bool compare_result(ov::TensorVector &rslts_cpu, ov::TensorVector &rslts_gpu, const float &T = 0.001f)
{
    bool bsimilar = true;
    for (size_t i = 0; i < rslts_cpu.size(); i++)
    {
        std::cout << "  == Comparing output idx = " << i << std::endl;
        float* cpu_data = rslts_cpu[i].data<float>();
        float* gpu_data = rslts_gpu[i].data<float>();
        for (size_t j = 0; j < rslts_cpu[i].get_size(); j++) {
            if (fabs(cpu_data[i] - gpu_data[i]) > T) {
                std::cout << "    rslt_cpu vs rslt_gpu [" << j << "], diff: " << cpu_data[j] << " vs " << gpu_data[j] << std::endl;
                bsimilar = false;
            }
        }
    }
    return bsimilar;
}

bool test_ov_model_with_custom_op()
{
    std::string device = "GPU";

    // Load model: from "python/custom_op/1_register_kernel/run.sh" test tmp folder.
    std::string ir = "../../python/custom_op/1_register_kernel/tmp/model_custom_op/export_ov_model/openvino_model_GPU_True.xml";
    std::string extension_path = "../../python/custom_op/1_register_kernel/cpu/build/libopenvino_custom_add_extension.so";
    std::string gpu_config = "../../python/custom_op/1_register_kernel/gpu/custom_add.xml";

    auto inp_shape = ov::Shape({2, 5,10});
    auto inp_data = randomData(inp_shape);
    auto inp_tensor = ov::Tensor(ov::element::f32, inp_shape);
    std::memcpy(inp_tensor.data(), inp_data.data(), inp_tensor.get_byte_size());

    auto rslts_cpu = test_model_device("CPU", ir, {inp_tensor}, extension_path, gpu_config);
    auto rslts_gpu = test_model_device("GPU", ir, {inp_tensor}, extension_path, gpu_config);

    // print result
    assert(rslts_cpu.size() == rslts_gpu.size());
    std::cout << "== Comparing CPU VS GPU results: " << std::endl;
    auto bsimilar = compare_result(rslts_cpu, rslts_gpu);
    std::cout << "  == Compare done: " << (bsimilar ? "Similar" : "Diff") << std::endl;
    return true;
}

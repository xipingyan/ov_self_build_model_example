#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <openvino/openvino.hpp> // 核心 OpenVINO 头文件

using namespace ov;

// 假设你的模型接受一个四维输入，例如 [1, Seq, Dim]
constexpr size_t BATCH_SIZE = 4;

/**
 * @brief OpenVINO C++ 推理示例
 *
 * @param model_path 模型 XML 文件路径
 * @param device_name 运行推理的设备，例如 "CPU", "GPU"
 */
void run_inference_example(const std::string& model_path, const std::string& device_name) {
    // --------------------------- 1. 初始化 Core 对象 ---------------------------
    Core core;
    std::cout << "OpenVINO Core initialized." << std::endl;

    core.add_extension("../cpu/build/libopenvino_custom_add_extension.so");
    if (device_name == "GPU") {
        core.set_property({{"CONFIG_FILE", "../gpu/custom_add.xml"}});
    }

    // --------------------------- 2. 读取模型 ------------------------------------
    std::shared_ptr<Model> model = core.read_model(model_path);
    std::cout << "Model '" << model->get_friendly_name() << "' loaded successfully." << std::endl;

    // 获取输入和输出的名称
    for (auto input : model->inputs())
    {
        std::string input_name = input.get_any_name();
        auto input_shape = input.get_partial_shape();

        std::cout << "  == Model Input Name: " << input_name << ", Shape: " << input_shape << ", Type: " << model->input().get_element_type() << std::endl;
    }
    for (auto output : model->outputs())
    {
        std::cout << "  == Model Output Name: " << output.get_any_name() << ", Shape: " << output.get_partial_shape() << ", Type: " << output.get_element_type() << std::endl;
    }

    // --------------------------- 4. 编译模型 ------------------------------------
    CompiledModel compiled_model = core.compile_model(model, device_name);
    std::cout << "Model compiled for device: " << device_name << std::endl;

    // --------------------------- 5. 创建推理请求 --------------------------------
    InferRequest infer_request = compiled_model.create_infer_request();
    std::cout << "Inference Request created." << std::endl;

    // --------------------------- 6. 准备输入数据 --------------------------------

    size_t inp_idx = 0;
    for (auto input : model->inputs())
    {        
        std::string input_name = input.get_any_name();
        auto input_pshape = input.get_partial_shape();
        auto data_type = input.get_element_type();

        // 假设输入数据是 float32 类型，形状为 [1, C, H, W]
        ov::Shape input_shape = ov::Shape({BATCH_SIZE, input_pshape.get_max_shape()[1], input_pshape.get_max_shape()[2]});
        std::cout << "  == input_shape = " << input_shape << std::endl;
        size_t total_input_size = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
        std::vector<float> input_data(total_input_size);
        // std::iota(input_data.begin(), input_data.end(), 0.0f);
        for (size_t i = 0; i < input_data.size(); i++)
        {
            input_data[i] = (i % 10) / 10.0f;
        }
        

        // 创建 OpenVINO 张量并绑定数据
        // ov::Tensor 的构造函数可以接受原始数据指针，实现零拷贝
        ov::Tensor input_tensor(data_type, input_shape, input_data.data());

        // 将输入张量放入推理请求中
        infer_request.set_input_tensor(inp_idx, input_tensor);
        inp_idx++;
    }

    // --------------------------- 7. 执行同步推理 --------------------------------
    infer_request.infer();
    std::cout << "Inference completed (Synchronous)." << std::endl;

    // --------------------------- 8. 获取结果 ------------------------------------
    const ov::Tensor& output_tensor = infer_request.get_output_tensor(0);
    const float* output_data = output_tensor.data<const float>();
    
    // 打印前几个输出结果 (假设输出是 float32)
    size_t output_size = output_tensor.get_size();
    size_t print_limit = std::min((size_t)10, output_size);
    
    std::cout << "First " << print_limit << " output values:" << std::endl;
    for (size_t i = 0; i < print_limit; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "\nOutput size: " << output_size << std::endl;

    // --------------------------- 9. 释放资源 (可选，Core 和 Model 对象超出作用域会自动释放) ---------------------------
    std::cout << "Example finished successfully." << std::endl;
}

int main(int argc, char* argv[]) {
    std::string device_name = "GPU";
    if (argc == 2) {
        device_name = argv[1];
    }
    else {
        std::cout << "$ app [dev name] " << std::endl;
    }
    std::cout << "== Device = " << device_name << std::endl;

    std::string workpath = "/mnt/xiping/mygithub/ov_self_build_model_example/python/custom_op/1_register_kernel";
    workpath = "../";
    std::string model_path = workpath + "/tmp/model_custom_op_2_outputs/export_ov_model/openvino_model_GPU_True.xml";

    run_inference_example(model_path, device_name);
    return 0;
}
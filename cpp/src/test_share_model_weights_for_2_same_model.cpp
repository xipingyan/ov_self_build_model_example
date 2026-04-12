#include "private.hpp"
#include "utils.hpp"

#include <fstream>
#include <sstream>

#include <openvino/runtime/tensor.hpp>

// Test case: share model weights for 2 same model
namespace
{
    std::string basename_of(const std::string& path)
    {
        const size_t pos = path.find_last_of('/');
        if (pos == std::string::npos) {
            return path;
        }
        return path.substr(pos + 1);
    }

    std::string read_file(const std::string& file_path)
    {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

    ov::Tensor read_bin_to_tensor(const std::string& bin_path, bool mmap = true)
    {
        // Uses OpenVINO's public API to read tensor data from file.
        // When mmap=true, the returned tensor is backed by file mapping and pages are loaded on demand.
        return ov::read_tensor_data(bin_path, ov::element::u8, ov::PartialShape::dynamic(1), 0, mmap);
    }

    // Read xml to string, bin to ov::Tensor.
    std::pair<std::string, ov::Tensor> load_xml_bin(std::string xml_path, std::string bin_path = std::string())
    {
        if (bin_path.empty()) {
            bin_path = xml_path;
            bin_path.replace(bin_path.find(".xml"), 4, ".bin");
        }
        std::string xml_str = read_file(xml_path);
        bool enable_mmap = false; // Set to true to enable memory mapping for the .bin file, which can reduce memory usage when loading large models.
        enable_mmap = true;
        ov::Tensor bin_tensor = read_bin_to_tensor(bin_path, enable_mmap);
        return {xml_str, bin_tensor};
    }

    // Print current process memory usage for debugging.
    // Note: RSS is *resident* RAM, and may be far smaller than a model .bin file size when OpenVINO uses mmap/lazy paging.
    void print_cur_vram(std::string prefix = "")
    {
        // 1) Quick summary from /proc/self/status.
        {
            std::ifstream status_file("/proc/self/status");
            if (!status_file.is_open()) {
                std::cerr << "Failed to open /proc/self/status" << std::endl;
            } else {
                std::string line;
                std::string vmrss;
                std::string rss_anon;
                std::string rss_file;
                std::string rss_shmem;
                while (std::getline(status_file, line)) {
                    if (line.rfind("VmRSS:", 0) == 0) vmrss = line.substr(std::string("VmRSS:").size());
                    else if (line.rfind("RssAnon:", 0) == 0) rss_anon = line.substr(std::string("RssAnon:").size());
                    else if (line.rfind("RssFile:", 0) == 0) rss_file = line.substr(std::string("RssFile:").size());
                    else if (line.rfind("RssShmem:", 0) == 0) rss_shmem = line.substr(std::string("RssShmem:").size());
                }
                if (!vmrss.empty()) {
                    std::cout << prefix << "VmRSS:" << vmrss;
                    if (!rss_anon.empty() || !rss_file.empty() || !rss_shmem.empty()) {
                        std::cout << " | RssAnon:" << rss_anon
                                  << " | RssFile:" << rss_file
                                  << " | RssShmem:" << rss_shmem;
                    }
                    std::cout << std::endl;
                    return;
                }
            }
        }
    }

    void infer(ov::InferRequest& infer_request)
    {
        auto input_ids = ov::Tensor(ov::element::i64, {1, 512});
        for (size_t i = 0; i < 512; ++i) {
            static_cast<int64_t*>(input_ids.data())[i] = static_cast<int64_t>(i + 1000);
        }
        infer_request.set_input_tensor(0, input_ids);

        infer_request.infer();
        auto output = infer_request.get_output_tensor(0);
        // std::cout << "Output tensor shape: " << output.get_shape() << std::endl;
        print_cur_vram("After inference: ");
    }
} // namespace

void test_case_1(std::string xml_path)
{
    std::cout << "---> test_case_1: share model weights for 2 same model" << std::endl;
    print_cur_vram("Before loading model: ");

    auto [xml_str, bin_tensor] = load_xml_bin(xml_path);
    print_cur_vram("After loading model: ");

    ov::Core core;
    core.get_versions("CPU");
    auto model1 = core.read_model(xml_str, bin_tensor);
    print_cur_vram("After reading model1: ");
    auto model2 = core.read_model(xml_str, bin_tensor);
    print_cur_vram("After reading model2: ");

    auto cm1 = core.compile_model(model1, "CPU");
    print_cur_vram("After compiling model1: ");
    auto cm2 = core.compile_model(model2, "CPU");
    print_cur_vram("After compiling model2: ");

    auto infer_request1 = cm1.create_infer_request();
    print_cur_vram("After creating infer request1: ");
    auto infer_request2 = cm2.create_infer_request();
    print_cur_vram("After creating infer request2: ");

    infer(infer_request1);
    infer(infer_request2);
}

void test_case_2(std::string xml_path)
{
    std::cout << "---> test_case_2: no share model weights for 2 same model" << std::endl;
    print_cur_vram("Before loading model: ");

    ov::Core core;
    // core.set_property(ov::enable_mmap(false));
    core.get_versions("CPU");
    auto model1 = core.read_model(xml_path);
    print_cur_vram("After reading model1: ");
    auto model2 = core.read_model(xml_path);
    print_cur_vram("After reading model2: ");

    auto cm1 = core.compile_model(model1, "CPU");
    print_cur_vram("After compiling model1: ");
    auto cm2 = core.compile_model(model2, "CPU");
    print_cur_vram("After compiling model2: ");

    auto infer_request1 = cm1.create_infer_request();
    print_cur_vram("After creating infer request1: ");
    auto infer_request2 = cm2.create_infer_request();
    print_cur_vram("After creating infer request2: ");

    infer(infer_request1);
    infer(infer_request2);
}

bool test_share_model_weights_for_2_same_model()
{
    // bin size: 372 MB.
    std::string xml_path = "../../../modular_genai/composable_pipeline/tests/test_models/Qwen3-Omni-4B-Instruct-multilingual-int4/openvino_text_embeddings_model.xml";
    test_case_1(xml_path);
    test_case_2(xml_path);
    return true;
}

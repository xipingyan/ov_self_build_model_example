#include "private.hpp"
#include "utils.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>

static bool g_enable_remote_tensor = std::getenv("ENABLE_RT") && std::getenv("ENABLE_RT") == std::string("1");

static std::shared_ptr<ov::Model> initModel(std::string model_name = "model_name")
{
    std::cout << " == initModel with model name: [" << model_name << "]" << std::endl;
    // Input 1: parameter
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1024, 1024});

    auto weights_shape = ov::Shape({1024, 1024});
    auto weights_data = randomData(weights_shape);
    auto weights = std::make_shared<ov::opset1::Constant>(ov::element::f32, weights_shape, weights_data);

    auto m = std::make_shared<ov::opset1::MatMul>(input1, weights, false, false);

    auto result = std::make_shared<ov::opset6::Result>(m);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input1}), model_name);

    return model;
}

// Check if exist device copy:
// ENABLE_RT=1 onetrace --chrome-call-logging --chrome-device-timeline ./testapp 
// onetrace --chrome-call-logging --chrome-device-timeline ./testapp 
bool test_remote_tensor()
{
    std::cout << "== test_remote_tensor =======================" << std::endl;
    std::cout << "    g_enable_remote_tensor = " << g_enable_remote_tensor << std::endl;
    if (!g_enable_remote_tensor)
        std::cout << "    export ENABLE_RT=1 to enable remote tensor." << std::endl;

    // OpenVINO Core
    ov::Core core = ov::Core();
    std::string device = "GPU";

    // Initial input
    ov::Tensor input1 = initTensor(ov::element::f32, ov::Shape{1024, 1024}, randomData(ov::Shape{1024, 1024}));

    auto model_1 = initModel("model_1");
    auto model_2 = initModel("model_2");

    // Compile model.
    std::cout << " == compile_model" << std::endl;
    auto compiledModel_1 = core.compile_model(model_1, device);
    auto compiledModel_2 = core.compile_model(model_2, device);

    std::cout << " == create_infer_request" << std::endl;
    auto inferRequest_1 = compiledModel_1.create_infer_request();
    auto inferRequest_2 = compiledModel_2.create_infer_request();

    ov::RemoteTensor remote_output_1;
    if (g_enable_remote_tensor)
    {
        ov::RemoteContext context = compiledModel_1.get_context();
        ov::Shape output_shape = inferRequest_1.get_output_tensor().get_shape();
        remote_output_1 = context.create_tensor(ov::element::f32, output_shape);
        inferRequest_1.set_output_tensor(0, remote_output_1);
    }

    // Inference
    std::cout << "  == Start infer. device=" << device << std::endl;
    for (auto i = 0; i < 2; i++)
    {
        std::cout << "  == Loop " << i << std::endl;
        inferRequest_1.set_input_tensors(0, {input1});
        inferRequest_1.infer();

        if (g_enable_remote_tensor)
        {
            inferRequest_2.set_input_tensors(0, {remote_output_1});
        }
        else
        {
            inferRequest_2.set_input_tensors(0, {inferRequest_1.get_output_tensor(0)});
        }
        inferRequest_2.infer();

        auto outTensor_2 = inferRequest_2.get_output_tensor();
        // Check output shape
        const auto &outShape = outTensor_2.get_shape();
        std::cout << "  == outShape = " << outShape << std::endl;
        std::cout << "  == outdata = " << outTensor_2.data<float>()[0] << std::endl;
    }

    std::cout << "== Done." << std::endl;
    return true;
}

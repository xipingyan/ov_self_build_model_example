#include "header.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>

// Construct OpenVINO subgraph
// Input1      const
//   \           |
//    \       convert
//     \         |
//      \    ReadValue
//       \      /
//        MatMul
//          |
//        Output
static std::shared_ptr<ov::Model> initStatefulModel()
{
    // Input 1: parameter
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape({1, 4}));

    // Input 2: const
    auto const1 = std::make_shared<ov::opset1::Constant>(ov::element::i32, ov::Shape({4, 1}), std::vector<int>({2, 2, 2, 2}));
    auto convert1 = std::make_shared<ov::opset1::Convert>(const1, ov::element::f32);

    // Stateful
    auto var_info = ov::op::util::VariableInfo({ov::Shape({4, 1}), ov::element::f32, std::string("variable0")});
    auto var = std::make_shared<ov::op::util::Variable>(var_info);
    auto rv = std::make_shared<ov::opset6::ReadValue>(convert1, var);
    auto assign = std::make_shared<ov::opset6::Assign>(rv, var);

    // Note: MatMul's result is wrong, but Multiply's result is right
    auto m = std::make_shared<ov::opset1::MatMul>(input1, rv, false, false);
    // auto m = std::make_shared<ov::opset1::Multiply>(input1, rv);

    auto result = std::make_shared<ov::opset6::Result>(m);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::SinkVector({assign}), ov::ParameterVector({input1}), "stateful_model");

    return model;
}

bool test_model_stateful_device(std::string device)
{
    bool bSync = true;
    // OpenVINO Core
    ov::Core core = ov::Core();

    // Initial input
    ov::Tensor input1 = ov::Tensor(ov::element::f32, ov::Shape({1, 4}));
    auto input1_data = std::vector<float>({1, 2, 3, 4});
    memcpy(input1.data<float>(), input1_data.data(), sizeof(float) * 4);

    // Init model
    auto model = initStatefulModel();

    // Compile model.
    auto compiledModel = core.compile_model(model, device);
    auto inferRequest = compiledModel.create_infer_request();

    inferRequest.set_input_tensors(0, {input1});
    inferRequest.reset_state();

    // Inference
    std::cout << "  == Start infer. device=" << device << std::endl;
    if (bSync)
    {
        inferRequest.infer();
    }
    else
    {
        inferRequest.start_async();
        inferRequest.wait();
    }

    std::cout << "  == Get output tensor." << std::endl;
    auto outTensor = inferRequest.get_output_tensor();

    // Check output shape
    const auto &outShape = outTensor.get_shape();
    std::cout << "  == outShape = " << outShape << std::endl;
    const float *ouptputData = outTensor.data<float>();
    std::cout << "  == ouptputData = [";
    for (size_t i = 0; i < outTensor.get_size(); i++)
    {
        std::cout << ouptputData[i] << ", ";
    }
    std::cout << "]\n";

    std::cout << "Test model finish." << std::endl;
    return true;
}

bool test_model_stateful() {
    test_model_stateful_device("CPU");
    test_model_stateful_device("TEMPLATE");
    return true;
}

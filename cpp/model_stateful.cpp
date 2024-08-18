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
static std::shared_ptr<ov::Model> initStatefulModelConstInput()
{
    std::cout << " == Init model: ReadValue inpupt with const" << std::endl;
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

// Construct OpenVINO subgraph
//           Input2
//             |
//          convert
//             |
//   Input1  ReadValue  Input3
//       \    /         /
//       MatMul  ReadValue
//          \     /
//            Add
//             |
//           Output
static std::shared_ptr<ov::Model> initStatefulModelVarInput()
{
    std::cout << " == Init model: ReadValue inpupt with variable" << std::endl;
    // Input 1: parameter
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape({1, 4}));

    // Input 2: Parameter
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({4, 1}));
    auto convert1 = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);

    // Input 3: parameter (bias)
    auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape({1}));

    // Stateful 1
    auto var_info = ov::op::util::VariableInfo({ov::Shape({4, 1}), ov::element::f32, std::string("rv_2")});
    auto var = std::make_shared<ov::op::util::Variable>(var_info);
    auto rv = std::make_shared<ov::opset6::ReadValue>(convert1, var);
    auto assign = std::make_shared<ov::opset6::Assign>(rv, var);

    // Note: MatMul's result is wrong, but Multiply's result is right
    auto m = std::make_shared<ov::opset1::MatMul>(input1, rv, false, false);
    // auto m = std::make_shared<ov::opset1::Multiply>(input1, rv);

    // Stateful 2
    auto var_info_2 = ov::op::util::VariableInfo({ov::Shape({1}), ov::element::f32, std::string("rv_3")});
    auto var_2 = std::make_shared<ov::op::util::Variable>(var_info_2);
    auto rv_2 = std::make_shared<ov::opset6::ReadValue>(input3, var_2);
    auto assign_2 = std::make_shared<ov::opset6::Assign>(rv_2, var_2);

    auto add = std::make_shared<ov::opset1::Add>(m, rv_2);
    auto result = std::make_shared<ov::opset6::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::SinkVector({assign, assign_2}),
                                             ov::ParameterVector({input1, input2, input3}), "stateful_model");

    return model;
}

bool test_model_stateful_device(std::string device)
{
    std::cout << " == test_model_stateful_device =======================" << std::endl;
    bool bSync = true;
    // OpenVINO Core
    ov::Core core = ov::Core();

#define ReadValueWithConstInput 0 // ReadValue with const input
    // Initial input
    ov::Tensor input1 = ov::Tensor(ov::element::f32, ov::Shape({1, 4}));
    auto input1_data = std::vector<float>({1, 2, 3, 4});
    memcpy(input1.data<float>(), input1_data.data(), sizeof(float) * 4);

    ov::Tensor input2 = ov::Tensor(ov::element::i32, ov::Shape({4, 1}));
    ov::Tensor input3 = ov::Tensor(ov::element::f32, ov::Shape({1}));

    // Init model
#if ReadValueWithConstInput
    auto model = initStatefulModelConstInput();
#else
    auto model = initStatefulModelVarInput();
#endif

    // Compile model.
    std::cout << " == compile_model" << std::endl;
    auto compiledModel = core.compile_model(model, device);
    std::cout << " == create_infer_request" << std::endl;
    auto inferRequest = compiledModel.create_infer_request();

    std::cout << " == set_input_tensors" << std::endl;
#if ReadValueWithConstInput
    inferRequest.set_input_tensors(0, {input1});
#else
    inferRequest.set_input_tensors(0, {input1});
    inferRequest.set_input_tensors(1, {input2});
    inferRequest.set_input_tensors(2, {input3});
#endif
    // std::cout << " == reset_state" << std::endl;
    // inferRequest.reset_state();

    // Inference
    std::cout << "  == Start infer. device=" << device << std::endl;
    for (auto i = 0; i < 3; i++)
    {
        std::cout << "  == Loop " << i + 1 << std::endl;

        // Update input2
        auto input2_data = std::vector<int>({i + 1, i + 1, i + 1, i + 1});
        memcpy(input2.data<int>(), input2_data.data(), sizeof(int) * 4);

        // Update input3
        auto input3_data = std::vector<float>({i + 1.0f});
        memcpy(input3.data<float>(), input3_data.data(), sizeof(float) * 1);

        // Reset state
        auto states = inferRequest.query_state();
        for (auto state : states)
        {
            auto name = state.get_name();
            std::cout << "   --> State:" << name;
            auto reset_state = [&]()
            {
                std::cout << ".reset()";
                state.reset();
            };
            if (name == "rv_2")
            {
                reset_state();
            }
            if (name == "rv_3")
            {
                reset_state();
            }
            std::cout << std::endl;
        }

        // Infer
        if (bSync)
        {
            inferRequest.infer();
        }
        else
        {
            inferRequest.start_async();
            inferRequest.wait();
        }

        auto outTensor = inferRequest.get_output_tensor();

        // Check output shape
        const auto &outShape = outTensor.get_shape();
        std::cout << "  == outShape = " << outShape;
        const float *ouptputData = outTensor.data<float>();
        std::cout << ", ouptputData = [";
        for (size_t i = 0; i < outTensor.get_size(); i++)
        {
            std::cout << ouptputData[i] << ", ";
        }
        std::cout << "]\n";
    }

    std::cout << "Test model finish." << std::endl;
    return true;
}

bool test_model_stateful()
{
    test_model_stateful_device("CPU");
    // test_model_stateful_device("TEMPLATE");
    return true;
}

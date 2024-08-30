#include "private.hpp"
#include "utils.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>

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
static std::shared_ptr<ov::Model> initIfModel()
{
    std::cout << " == Init model: ReadValue inpupt with const" << std::endl;
    // Input 1: parameter
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({1}));
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({1}));
    auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({1}));

    // If
    auto if_op = std::make_shared<ov::opset8::If>();

    // If then
    auto inp_then = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({1}));
    auto const_val_2 = std::make_shared<ov::opset1::Constant>(ov::element::i32, ov::Shape({1}), std::vector<int>({2}));
    auto mulitply_then = std::make_shared<ov::opset1::Multiply>(inp_then, const_val_2);
    auto result_then = std::make_shared<ov::opset6::Result>(mulitply_then);
    auto model_then = std::make_shared<ov::Model>(ov::ResultVector({result_then}), ov::ParameterVector({inp_then}), "model_then");

    // If else
    auto inp_else = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({1}));
    auto const_val_4 = std::make_shared<ov::opset1::Constant>(ov::element::i32, ov::Shape({1}), std::vector<int>({4}));
    auto mulitply_else = std::make_shared<ov::opset1::Multiply>(inp_else, const_val_4);
    auto result_else = std::make_shared<ov::opset6::Result>(mulitply_else);
    auto model_else = std::make_shared<ov::Model>(ov::ResultVector({result_else}), ov::ParameterVector({inp_else}), "model_else");

    if_op->set_then_body(model_then);
    if_op->set_else_body(model_else);

    if_op->set_input(input1->output(0), inp_then, inp_else);
    if_op->set_input(input2->output(0), inp_then, nullptr);
    if_op->set_input(input3->output(0), nullptr, inp_else);

    if_op->set_output(result_then, result_else);

    auto result = std::make_shared<ov::opset6::Result>(if_op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input1, input2, input3}), "model_if");

    return model;
}

bool test_model_if_device(std::string device)
{
    std::cout << " == Start test: " << __FUNCTION__ << " =======================" << std::endl;
    ov::Core core = ov::Core();

    // Initial input
    ov::Tensor input1 = initTensor<int>(ov::element::i32, ov::Shape({1}), std::vector<int>({0}));
    ov::Tensor input2 = initTensor<int>(ov::element::i32, ov::Shape({1}), std::vector<int>({2}));
    ov::Tensor input3 = initTensor<int>(ov::element::i32, ov::Shape({1}), std::vector<int>({4}));

    auto model = initIfModel();

    // Compile model.
    std::cout << " == compile_model" << std::endl;
    auto compiledModel = core.compile_model(model, device);
    std::cout << " == create_infer_request" << std::endl;
    auto inferRequest = compiledModel.create_infer_request();

    std::cout << " == set_input_tensors" << std::endl;
    inferRequest.set_input_tensors(0, {input1});
    inferRequest.set_input_tensors(1, {input2});
    inferRequest.set_input_tensors(2, {input3});

    // Inference
    std::cout << "  == Start infer. device=" << device << std::endl;
    inferRequest.infer();

    auto outTensor = inferRequest.get_output_tensor();

    // Check output
    const int *ouptputData = outTensor.data<int>();
    std::cout << "  == outShape = " << outTensor.get_shape() << ", ouptputData= " << ouptputData[0] << std::endl;
    std::cout << "Test model finish." << std::endl;
    return true;
}

bool test_model_if()
{
    test_model_if_device("CPU");
    // test_model_if_device("TEMPLATE");
    return true;
}

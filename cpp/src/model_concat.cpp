#include "private.hpp"
#include "utils.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>


static std::shared_ptr<ov::Model> initConcatModel()
{
    std::cout << " == Init:" << __FUNCTION__ << std::endl;
    // Input 1: parameter
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({1, 2, 3}));
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape({1, 2, 3}));

    int64_t concat_axis = 0;
    ov::NodeVector inputs_to_concat = {input1, input2};
    auto temporal_images = std::make_shared<ov::op::v0::Concat>(inputs_to_concat, concat_axis);

    auto result = std::make_shared<ov::opset6::Result>(temporal_images);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input1, input2}), "model_concat");

    return model;
}

static bool test_model_device(std::string device)
{
    std::cout << " == Start test: =========== " << __FUNCTION__ << " ============" << std::endl;
    ov::Core core = ov::Core();

    auto inp_data_1 = std::vector<int>({10, 11, 12, 13, 14, 15});
    auto inp_data_2 = std::vector<int>({20, 21, 22, 23, 24, 25});

    // Initial input
    ov::Tensor input1 = initTensor<int>(ov::element::i32, ov::Shape({1, 2, 3}), inp_data_1);
    ov::Tensor input2 = initTensor<int>(ov::element::i32, ov::Shape({1, 2, 3}), inp_data_2);
    // ov::Tensor input2 = ov::Tensor(ov::element::i32, ov::Shape({1, 2, 3}), input1.data<int32_t>());

    auto model = initConcatModel();

    // Compile model.
    std::cout << " == compile_model with device: " << device << std::endl;
    auto compiledModel = core.compile_model(model, device);
    std::cout << " == create_infer_request" << std::endl;
    auto inferRequest = compiledModel.create_infer_request();

    std::cout << " == set_input_tensors" << std::endl;
    inferRequest.set_input_tensors(0, {input1});
    inferRequest.set_input_tensors(1, {input2});

    // Inference
    std::cout << "  == Start infer. device=" << device << std::endl;
    inferRequest.infer();

    auto outTensor = inferRequest.get_output_tensor();

    // Check output
    const int *ouptputData = outTensor.data<int>();
    std::cout << "  == outShape = " << outTensor.get_shape() << ", ouptputData[0] = " << ouptputData[0] << std::endl;
    std::cout << "    == All results: ";
    for (int i = 0; i < outTensor.get_size(); i++) {
        std::cout << ouptputData[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "  == Test model finish." << std::endl;
    return true;
}

bool test_model_concat()
{
    test_model_device("CPU");
    test_model_device("GPU");
    return true;
}

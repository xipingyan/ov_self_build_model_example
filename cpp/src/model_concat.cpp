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
    // ov::Tensor input2 = initTensor<int>(ov::element::i32, ov::Shape({1, 2, 3}), inp_data_2);
    ov::Tensor input2 = ov::Tensor(ov::element::i32, ov::Shape({1, 2, 3}), input1.data<int32_t>());

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

std::shared_ptr<ov::Node> create_f32_nchw_input(std::shared_ptr<ov::Node> input) {
    auto raw_images_f32 = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    auto img_trans = std::make_shared<ov::op::v1::Transpose>(
        raw_images_f32,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));
    return img_trans;
}


std::shared_ptr<ov::Node> create_bicubic_resize(std::shared_ptr<ov::Node> input,
                                                const std::shared_ptr<ov::Node>& target_size) {
    // Create axes for height and width dimensions (assuming NCHW layout)
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});

    // Configure interpolation attributes for bicubic resize
    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.cube_coeff = -0.75f;  // Standard bicubic coefficient
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = false;

    // Create interpolate operation
    auto interpolate = std::make_shared<ov::op::v11::Interpolate>(input, target_size, axes, attrs);

    return interpolate;
}

std::shared_ptr<ov::Node> create_normalization(std::shared_ptr<ov::Node> input,
                                               const std::shared_ptr<ov::Node>& mean,
                                               const std::shared_ptr<ov::Node>& std) {
    // clamp to 0 ~ 255
    auto image_clamp = std::make_shared<ov::op::v0::Clamp>(input, 0, 255);
    // Subtract mean
    auto mean_subtracted = std::make_shared<ov::op::v1::Subtract>(image_clamp, mean);

    // Divide by std
    auto normalized = std::make_shared<ov::op::v1::Multiply>(mean_subtracted, std);

    return normalized;
}

static std::shared_ptr<ov::Model> initSubgrap() {
    auto raw_images_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});
    auto raw_images_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});

    auto resize_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});

    std::vector<float> a_image_mean(3, 0);
    std::vector<float> a_image_scale(3, 1.0f);

    // Note: Constant node can't construct from ov::Tensor.
    ov::op::v0::Constant image_mean_const(ov::element::f32, ov::Shape{1, a_image_mean.size(), 1, 1}, a_image_mean.data());
    ov::op::v0::Constant image_scale_const(ov::element::f32, ov::Shape{1, a_image_scale.size(), 1, 1}, a_image_scale.data());

    auto img_f32_nchw_1 = create_f32_nchw_input(raw_images_1);
    auto img_resized_1 = create_bicubic_resize(img_f32_nchw_1, resize_shape);
    auto img_normalized_1 = create_normalization(img_resized_1, std::make_shared<ov::op::v0::Constant>(image_mean_const), std::make_shared<ov::op::v0::Constant>(image_scale_const));

    auto img_f32_nchw_2 = create_f32_nchw_input(raw_images_2);
    auto img_resized_2 = create_bicubic_resize(img_f32_nchw_2, resize_shape);
    auto img_normalized_2 = create_normalization(img_resized_2, std::make_shared<ov::op::v0::Constant>(image_mean_const), std::make_shared<ov::op::v0::Constant>(image_scale_const));
    int64_t concat_axis = 0;
    ov::OutputVector inputs_to_concat = {img_normalized_1->output(0), img_normalized_2->output(0)};
    auto temporal_images = std::make_shared<ov::op::v0::Concat>(inputs_to_concat, concat_axis);

    auto result = std::make_shared<ov::opset6::Result>(temporal_images);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({raw_images_1, raw_images_2, resize_shape}), "model_concat");

    return model;
}

static std::vector<float> test_subgraph_model_device(std::string device, ov::Shape data_shape, std::vector<uint8_t> &inp_data_1)
{
    std::cout << "== Start test: =========== " << __FUNCTION__ << " ============" << std::endl;
    ov::Core core = ov::Core();

    // Initial input
    ov::Tensor raw_images_1 = initTensor<uint8_t>(ov::element::u8, data_shape, inp_data_1);
    ov::Tensor raw_images_2 = initTensor<uint8_t>(ov::element::u8, data_shape, inp_data_1);
    // ov::Tensor raw_images_2 = ov::Tensor(ov::element::u8, data_shape, raw_images_1.data<uint8_t>());
    ov::Tensor resize_shape = initTensor<int64_t>(ov::element::i64, ov::Shape({2}), std::vector<int64_t>{2,2});

    auto model = initSubgrap();

    // Compile model.
    std::cout << " == compile_model with device: " << device << std::endl;
    auto compiledModel = core.compile_model(model, device);
    std::cout << " == create_infer_request" << std::endl;
    auto inferRequest = compiledModel.create_infer_request();

    std::cout << " == set_input_tensors" << std::endl;
    inferRequest.set_input_tensor(0, raw_images_1);
    inferRequest.set_input_tensor(1, raw_images_2);
    inferRequest.set_input_tensor(2, resize_shape);

    // Inference
    std::cout << " == Start infer. device=" << device << std::endl;
    inferRequest.infer();

    auto outTensor = inferRequest.get_output_tensor(0);

    auto print_tensor = [](const ov::Tensor& out_tensor) {
        const float *ouptputData = out_tensor.data<float>();
        std::cout << "  == output shape = " << out_tensor.get_shape() << ", " << out_tensor.get_element_type() << std::endl;
        std::cout << "    == All results: ";
        std::vector<float> vec_rslt;
        for (int i = 0; i < out_tensor.get_size(); i++) {
            std::cout << ouptputData[i] << ", ";
            vec_rslt.push_back(ouptputData[i]);
        }
        std::cout << std::endl;
        return vec_rslt;
    };

    // Check output
    auto vec_rslt = print_tensor(outTensor);

    std::cout << "  == Test model finish." << std::endl;
    return vec_rslt;
}

bool test_model_concat()
{
    // test_model_device("CPU");
    // test_model_device("GPU");

    auto data_shape = ov::Shape({1, 4, 6, 3});
    auto inp_data_1 = randomData_U8(data_shape);
    auto rslt_cpu = test_subgraph_model_device("CPU", data_shape, inp_data_1);
    auto rslt_gpu = test_subgraph_model_device("GPU", data_shape, inp_data_1);
    // print result
    assert(rslt_cpu.size() == rslt_gpu.size());
    std::cout << "== Comparing CPU VS GPU results: " << std::endl;
    bool bsimilar = true;
    for (size_t i = 0; i < rslt_cpu.size(); i++)
    {
        if (fabs(rslt_cpu[i] - rslt_gpu[i]) > 0.2f) {
            std::cout << "    rslt_cpu vs rslt_gpu [" << i << "], diff: " << rslt_cpu[i] << " vs " << rslt_gpu[i] << std::endl;
            bsimilar = false;
        }
    }
    std::cout << "   Compare done: " << (bsimilar ? "Similar" : "Diff") << std::endl;

    return true;
}

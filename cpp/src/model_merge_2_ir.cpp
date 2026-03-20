#include "private.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/op/slice.hpp>

// TODO: 
// 0: Prepare 1 input: Input1 with shape {-1, FeaDim} and type f32. The data will be fed in runtime.
// 1: Draft 2 models;
// 2: Merge 2 models into 1 model. 
//      Get first model's output's last dimension, concat it to Input1.
//      Take Input1 as input of second model, and get the final output.

#define FeaDim 2

std::shared_ptr<ov::Model> draft_model() {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, FeaDim});
    auto weights = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{FeaDim, FeaDim}, std::vector<float>({2, 2, 2, 2}));
    auto m = std::make_shared<ov::opset1::MatMul>(input1, weights, false, false);
    auto result = std::make_shared<ov::opset6::Result>(m);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input1}), "model_1");
    return model;
}

std::shared_ptr<ov::Model> merge_2_models(const std::shared_ptr<ov::Model>& model_1, const std::shared_ptr<ov::Model>& model_2) {
    auto model_1_input = model_1->get_parameters().at(0);
    auto model_2_input = model_2->get_parameters().at(0);

    auto model_1_output = model_1->get_results()[0]->input_value(0);

    // Get last feature column of model_1's output, and concat it to model_1's input, then replace model_2's input with the concat result.
    // For example: if model_1 output shape is {3, 2}, after gather shape will be {3, 1}
    auto gather = std::make_shared<ov::opset1::Gather>(
        model_1_output,
        ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {-1}),
        ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    
    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{model_1_input->output(0), gather->output(0)}, 0);
    model_2_input->output(0).replace(concat->output(0));

    return std::make_shared<ov::Model>(model_2->get_results(), ov::ParameterVector{model_1_input}, "merged_model");
}

// common case, support to merge more mdoels.
std::shared_ptr<ov::Model> merge_2_models(const std::vector<std::shared_ptr<ov::Model>>& models) {
    assert(models.size() >= 2);

    auto model_1_input = models.front()->get_parameters().at(0);
    for (size_t i = 0; i < models.size() - 1; i++) {
        auto model_2_input = models[i + 1]->get_parameters().at(0);

        auto model_1_output = models[i]->get_results()[0]->input_value(0);

        auto start = ov::opset1::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
        auto stop = ov::opset1::Constant::create(ov::element::i32, ov::Shape{1}, {std::numeric_limits<int32_t>::max()});
        auto step = ov::opset1::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        // 4. 定义 Axes: 操作第一维度 (Axis 0)
        auto axes = ov::opset1::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        // 5. 创建 Slice 节点
        auto slice_node = std::make_shared<ov::opset8::Slice>(model_1_output, start, stop, step, axes);

        auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{model_1_input->output(0), slice_node->output(0)}, 1);
        model_2_input->output(0).replace(concat->output(0));
    }

    return std::make_shared<ov::Model>(models.back()->get_results(), ov::ParameterVector{models.front()->get_parameters().at(0)}, "merged_model");
}

bool test_merge_2_ov_ir() {
    auto model_1 = draft_model();
    auto model_2 = draft_model();
    auto merged_model = merge_2_models(model_1, model_2);

    ov::Core core = ov::Core();
    auto compiledModel = core.compile_model(merged_model, "CPU");
    auto inferRequest = compiledModel.create_infer_request();

    ov::Tensor input1 = initTensor<float>(ov::element::f32, ov::Shape{1, FeaDim}, std::vector<float>({1, 1, 1, 1}));
    inferRequest.set_input_tensors(0, {input1});
    inferRequest.infer();
    auto outTensor = inferRequest.get_output_tensor();
    const auto &outShape = outTensor.get_shape();
    std::cout << "  == outShape = " << outShape << std::endl;

    // dump ir.
    ov::serialize(merged_model, "merged_model.xml", "merged_model.bin");
    std::cout << "  == merged model serialize done. " << std::endl;

    return true;
}
#include "private.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/op/slice.hpp>

// TODO: 
// 0: For Input1, shape [?, 2].
// 1: Draft 2 same models;
// 2: Merge 2 models into 1 model. 
//      2.1: Get model_1's output, and get last element for axis 0.
//      2.2: Concat the last element with model_1's input, and replace model_2's input with the concat result.

#define FeaDim 2

// draft a model with one matmul op, and input shape is [?, 2].
// -----------------------------------------------
//  Input   Weigths
//      \    /
//      MatMul
//         |
//      Output
std::shared_ptr<ov::Model> draft_model(std::string matmal_name = "mm") {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, FeaDim});
    auto weights = createConstant<float>(ov::element::f32, ov::Shape{FeaDim, FeaDim}, std::vector<float>({2, 2, 2, 2}));
    auto m = std::make_shared<ov::opset1::MatMul>(input1, weights, false, false);
    auto result = std::make_shared<ov::opset6::Result>(m);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input1}), "model_1");
    return model;
}

std::pair<std::shared_ptr<ov::Model>, ov::Output<ov::Node>> merge_2_models(const std::shared_ptr<ov::Model> &model_1,
                                                                           const std::shared_ptr<ov::Model> &model_2,
                                                                           ov::Output<ov::Node> concat_input_1 = ov::Output<ov::Node>())
{
    auto model_1_input = model_1->get_parameters().at(0);
    if (concat_input_1.get_node_shared_ptr() == nullptr) {
        concat_input_1 = model_1_input->output(0);
    }
    auto model_2_input = model_2->get_parameters().at(0);

    auto model_1_output = model_1->get_results()[0]->input_value(0);

    // Get last element for axis 0.
    auto start = createConstant<int64_t>(ov::element::i64, ov::Shape{1}, {-1});
    auto stop = createConstant<int64_t>(ov::element::i64, ov::Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto step = createConstant<int64_t>(ov::element::i64, ov::Shape{1}, {1});
    auto axes = createConstant<int64_t>(ov::element::i64, ov::Shape{1}, {0});
    auto slice_node = std::make_shared<ov::opset8::Slice>(model_1_output, start, stop, step, axes);

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{concat_input_1, slice_node->output(0)}, 0);
    model_2_input->output(0).replace(concat->output(0));

    return {std::make_shared<ov::Model>(model_2->get_results(), ov::ParameterVector{model_1_input}, "merged_model"), concat->output(0)};
}

// common case, support to merge more mdoels.
std::shared_ptr<ov::Model> merge_models(const std::vector<std::shared_ptr<ov::Model>>& models) {
    assert(models.size() >= 2);

    ov::Output<ov::Node> concat_input_1 = models[0]->get_parameters().at(0)->output(0);
    std::shared_ptr<ov::Model> merged_model;
    std::tie(merged_model, concat_input_1) = merge_2_models(models[0], models[1], concat_input_1);
    for (size_t i = 2; i < models.size(); i++) {
        std::tie(merged_model, concat_input_1) = merge_2_models(merged_model, models[i], concat_input_1);
    }

    return merged_model;
}

bool test_merge_2_ov_ir() {
    auto model_1 = draft_model("mm1");
    auto model_2 = draft_model("mm2");
    // auto merged_model = merge_2_models(model_1, model_2);
    auto merged_model = merge_models({model_1, model_2, draft_model("mm3")});

    // Expected merged model:
    // -----------------------------------------------
    //   Input1[1,2]   Weigths1
    //    |   \      /
    //    |   MatMul1 [1,2]
    //    |    |
    //    |   Slice [1,2]
    //    |   /
    //   Concat [2,2]  Weigths2
    //    |   \       /
    //    |    MatMul2
    //    |     | [2,2]
    //    |   Slice
    //    |    / [1,2]
    //    Concat   Weigths3
    //  [3,2] \    /
    //        MatMul3
    //           |
    //         Output

    ov::Core core = ov::Core();
    auto compiledModel = core.compile_model(merged_model, "CPU");
    auto inferRequest = compiledModel.create_infer_request();

    ov::Tensor input1 = initTensor<float>(ov::element::f32, ov::Shape{1, FeaDim}, std::vector<float>({1, 1}));
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
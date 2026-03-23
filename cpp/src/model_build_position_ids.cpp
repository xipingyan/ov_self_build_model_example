#include "private.hpp"

#include <openvino/openvino.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/unsqueeze.hpp>

// TODO:
// 0: Build a model with position ids.
// 1: For input_embeds with shape [batch_size, token_num, feature_dim], output position_ids with shape [batch_size, token_num].
// 2: For position_ids, each row is [0, 1, 2, ..., token_num-1].
ov::Output<ov::Node> build_position_ids(const ov::Output<ov::Node> &inputs_embeds)
{
    auto embeds_shape = std::make_shared<ov::op::v3::ShapeOf>(inputs_embeds);
    auto batch_dim =
        std::make_shared<ov::op::v8::Gather>(embeds_shape,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), // indices 0
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0})  // axis 0
        );
    auto token_num = std::make_shared<ov::op::v8::Gather>(embeds_shape,
                                                          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), // indices 1
                                                          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0})  // axis 0
    );

    const int start_value = 0;

    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {start_value});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    ov::Output<ov::Node> stop;
    if (start_value == 0)
    {
        // Range stop is exclusive. With start=0, use stop=token_num to keep length == token_num.
        stop = token_num;
    }
    else
    {
        // Range stop is exclusive. With start=1, use stop=token_num+start to keep length == token_num.
        stop = std::make_shared<ov::op::v1::Add>(token_num, start);
    }
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);

    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto batch_dim_1d = std::make_shared<ov::op::v0::Unsqueeze>(batch_dim, axis0);
    auto layer_tokens_1d = std::make_shared<ov::op::v0::Unsqueeze>(token_num, axis0);
    auto target_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch_dim_1d, layer_tokens_1d}, 0);

    auto position_ids = std::make_shared<ov::op::v3::Broadcast>(
        range,
        target_shape,
        ov::op::BroadcastType::NUMPY);

    return position_ids;
}

std::shared_ptr<ov::Model> build_model_with_position_ids(const int &FEA_DIM)
{
    auto inputs_embeds = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, FEA_DIM});
    auto position_ids = build_position_ids(inputs_embeds);
    auto result = std::make_shared<ov::opset1::Result>(position_ids);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{inputs_embeds}, "position_ids_model");
}

bool test_build_position_ids()
{
    const int FEA_DIM = 4;
    auto model = build_model_with_position_ids(FEA_DIM);

    ov::Core core = ov::Core();
    auto compiledModel = core.compile_model(model, "CPU");
    auto inferRequest = compiledModel.create_infer_request();

    ov::Tensor input = initTensorWithValue0<float>(ov::element::f32, ov::Shape{2, 4, FEA_DIM});
    inferRequest.set_input_tensors(0, {input});
    inferRequest.infer();
    auto outTensor = inferRequest.get_output_tensor();
    const auto &outShape = outTensor.get_shape();
    std::cout << "  == outShape = " << outShape << std::endl;
    std::cout << "  == outData = \n";
    for (size_t i = 0; i < outTensor.get_size(); i++)
    {
        std::cout << "    outTensor[" << i << "] = " << outTensor.data<int64_t>()[i] << std::endl;
    }

    // dump ir.
    ov::serialize(model, "model.xml", "model.bin");
    std::cout << "  == model serialize done. " << std::endl;

    return true;
}
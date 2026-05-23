#include "private.hpp"
#include "utils.hpp"
#include "profiler.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>

// Test: remote tensor with dynamic shapes between two models.
// Model1 output (dynamic batch) feeds directly into Model2 input as remote tensor,
// avoiding device-to-host copy.
//
// Key finding:
//   context.create_tensor(type, Shape{0, N}) can create an empty remote tensor.
//   After set_output_tensor(), runtime will auto-reallocate to actual shape during infer.
//   The binding (infer_req1.output -> shared_tensor -> infer_req2.input) remains valid
//   across different batch sizes without re-binding.
//
// Usage:
//   ENABLE_RT=1 ./testapp    # with remote tensor (fast, no D2H copy between models)
//   ./testapp                # without remote tensor (slow, D2H + H2D between models)

static bool g_enable_rt = std::getenv("ENABLE_RT") && std::getenv("ENABLE_RT") == std::string("1");

// Model1: input[batch, 512] -> MatMul -> Softmax -> output[batch, 256]
static std::shared_ptr<ov::Model> createModel1()
{
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 512});
    auto weights_data = randomData(ov::Shape{512, 256}, 0.1f, 0.9f, 100);
    auto weights = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{512, 256}, weights_data);
    auto matmul = std::make_shared<ov::opset1::MatMul>(input, weights, false, false);
    auto softmax = std::make_shared<ov::opset1::Softmax>(matmul);
    auto result = std::make_shared<ov::opset6::Result>(softmax);
    return std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input}), "model_1_dynamic");
}

// Model2: input[batch, 256] -> MatMul -> Softmax -> output[batch, 128]
static std::shared_ptr<ov::Model> createModel2()
{
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
    auto weights_data = randomData(ov::Shape{256, 128}, 0.1f, 0.9f, 200);
    auto weights = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{256, 128}, weights_data);
    auto matmul = std::make_shared<ov::opset1::MatMul>(input, weights, false, false);
    auto softmax = std::make_shared<ov::opset1::Softmax>(matmul);
    auto result = std::make_shared<ov::opset6::Result>(softmax);
    return std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input}), "model_2_dynamic");
}

bool test_remote_tensor_dynamic()
{
    std::cout << "== test_remote_tensor_dynamic =======================" << std::endl;
    std::cout << "   g_enable_rt = " << g_enable_rt << std::endl;

    ov::Core core;
    std::string device = "GPU";

    auto model1 = createModel1();
    auto model2 = createModel2();

    // Compile with shared context for remote tensor
    auto compiled1 = core.compile_model(model1, device);
    ov::RemoteContext context = compiled1.get_context();
    ov::CompiledModel compiled2;
    if (g_enable_rt) {
        compiled2 = core.compile_model(model2, context);
    } else {
        compiled2 = core.compile_model(model2, device);
    }

    auto infer_req1 = compiled1.create_infer_request();
    auto infer_req2 = compiled2.create_infer_request();

    // Create an empty remote tensor with Shape{0, 256} and bind it as:
    //   infer_req1 output -> shared_tensor -> infer_req2 input
    // Runtime auto-reallocates to actual shape on each infer, binding stays valid.
    ov::Tensor shared_tensor;

    if (g_enable_rt) {
        shared_tensor = context.create_tensor(ov::element::f32, ov::Shape{0, 256});
        infer_req1.set_output_tensor(0, shared_tensor);
        infer_req2.set_input_tensor(0, shared_tensor);
    }

    // Test with different batch sizes to verify dynamic shape support
    std::vector<size_t> batch_sizes = {1, 4, 8, 2};

    for (auto batch : batch_sizes) {
        std::cout << "\n  -- batch = " << batch << std::endl;
        PROFILE(P, "batch_" + std::to_string(batch));

        // Create input tensor with current batch size
        ov::Shape input_shape = {batch, 512};
        ov::Tensor input_tensor = initTensor(ov::element::f32, input_shape, randomData(input_shape, 0.1f, 0.9f));

        // Infer model1
        infer_req1.set_input_tensor(0, input_tensor);
        infer_req1.infer();
        auto output1 = infer_req1.get_output_tensor(0);
        std::cout << "     model1 output shape: " << output1.get_shape()
                  << ", is_remote: " << output1.is<ov::RemoteTensor>() << std::endl;

        // model2 input is already bound to model1's output (shared_tensor), just infer
        if (!g_enable_rt) {
            infer_req2.set_input_tensor(0, output1);
        }
        infer_req2.infer();
        auto output2 = infer_req2.get_output_tensor(0);
        std::cout << "     model2 output shape: " << output2.get_shape()
                  << ", is_remote: " << output2.is<ov::RemoteTensor>() << std::endl;

        // Copy final result to host for verification
        ov::Tensor host_output = output2;
        if (output2.is<ov::RemoteTensor>()) {
            host_output = ov::Tensor(ov::element::f32, output2.get_shape());
            output2.copy_to(host_output);
        }

        std::cout << "     final output[0] = " << host_output.data<float>()[0] << std::endl;
        EXPECT_EQ(host_output.get_shape()[0], batch);
        EXPECT_EQ(host_output.get_shape()[1], 128);
    }

    std::cout << "\n== test_remote_tensor_dynamic Done." << std::endl;
    return true;
}

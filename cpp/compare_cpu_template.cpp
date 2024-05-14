#include "header.hpp"

#include <openvino/core/model.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
// #include <openvino/reference/multiply.hpp>

#ifndef EXPECT_EQ
#define EXPECT_EQ(a, b)                                                                      \
    if ((a) != (b))                                                                          \
    {                                                                                        \
        std::cout << "Line:" << __LINE__ << ", Failed: " << #a << " != " << #b << std::endl; \
        std::cout << "  [" << a << " != " << b << "]" << std::endl;                          \
        exit(0);                                                                             \
    }
#endif

// Construct OpenVINO subgraph
//      Input
//        |
//      Convert
//        |
//      Output
static std::shared_ptr<ov::Model> initModel(ov::element::Type rtPrc, ov::Shape inpShape)
{
    auto params1 = std::make_shared<ov::opset1::Parameter>(rtPrc, inpShape);
    auto cvt = std::make_shared<ov::opset1::Convert>(params1, ov::element::f32);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{cvt}, ov::ParameterVector{params1});
    return model;
}

ov::Tensor get_infer_result(std::string dev, ov::Tensor &input, ov::Core core, std::shared_ptr<ov::Model> model)
{
    // Compile model.
    auto compiled_model = core.compile_model(model, dev);
    auto infer_req = compiled_model.create_infer_request();

    infer_req.set_input_tensors(0, {input});

    bool bSync = true;
    if (bSync)
    {
        infer_req.infer();
    }
    else
    {
        infer_req.start_async();
        infer_req.wait();
    }

    return infer_req.get_output_tensor();
}

// Initialize input with all prob values of u8
ov::Tensor init_input()
{
#if 0
#define shape_sz 128
    ov::Shape inpShape = ov::Shape{shape_sz};
    uint8_t input_arr[shape_sz / 2] = {69, 65, 41, 101, 135, 97, 129, 97, 41, 41, 37,
                                       129, 69, 97, 37, 33, 5, 37, 73, 101, 73, 67, 33, 69, 39, 103, 135, 37, 3, 69,
                                       101, 35, 105, 65, 99, 129, 73, 3, 129, 137, 99, 33, 39, 37, 69, 131, 37, 133,
                                       105, 133, 101, 41, 97, 9, 39, 133, 5, 39, 9, 105, 5, 135, 103, 3};
    ov::Tensor input = ov::Tensor(ov::element::i4, inpShape, input_arr);
#else
    ov::Tensor input = ov::Tensor(ov::element::i4, ov::Shape{256*2});
    uint8_t *pdata =reinterpret_cast<uint8_t*>(input.data());
    for (size_t i = 0; i < 256; i++)
    {
        pdata[i] = static_cast<uint8_t>(i);
    }
#endif
    return input;
}

bool test_cpu_template_compare()
{
    ov::element::Type rtPrc = ov::element::i4;
    std::shared_ptr<ov::Model> model;
    ov::Tensor outTensor;

    // OpenVINO Core
    ov::Core core = ov::Core();

    // Initial input Tensors
    auto input = init_input();

    // Construct model(subgraph)
    model = initModel(rtPrc, input.get_shape());

    auto output_tensor_cpu = get_infer_result("CPU", input, core, model);
    // Enable template plugin: -DENABLE_TEMPLATE_REGISTRATION=ON
    auto output_tensor_ref = get_infer_result("TEMPLATE", input, core, model);

    // Check output shape
    EXPECT_EQ(output_tensor_cpu.get_shape(), output_tensor_ref.get_shape());

    // Check output data
    bool expectEQZero = false;
    const float *ref_data = output_tensor_ref.data<float>();
    const float *cpu_data = output_tensor_cpu.data<float>();
    for (size_t i = 0; i < output_tensor_cpu.get_size(); i++)
    {
        EXPECT_EQ(ref_data[i], cpu_data[i]);
    }

    std::cout << "  Test model pass." << std::endl;
    return true;
}

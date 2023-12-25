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

void initBuf(float *pConst, size_t sz, float value)
{
    for (size_t i = 0; i < sz; ++i)
    {
        pConst[i] = value;
    }
}

// Construct OpenVINO subgraph
// Input1   Input2
//     \     /
//      \   /
//       \ /
//     Multiply
//        |
//      Output
std::shared_ptr<ov::Model> initModel(ov::element::Type rtPrc, ov::Shape inpShape)
{
    auto params1 = std::make_shared<ov::opset1::Parameter>(rtPrc, inpShape);
    auto params2 = std::make_shared<ov::opset1::Parameter>(rtPrc, inpShape);
    auto mul = std::make_shared<ov::opset1::Multiply>(params1, params2);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{params1, params2});
    return model;
}

bool test_model_multiply()
{
    bool bSync = true;
    ov::Shape inpShape = ov::Shape{1, 5, 12, 64};
    ov::element::Type rtPrc = ov::element::f32;
    std::vector<float *> vecConst;
    for (int i = 0; i < 3; i++)
    {
        vecConst.emplace_back(new float[shape_size(inpShape)]);
    }

    std::shared_ptr<ov::Model> model;
    ov::Tensor outTensor;

    // OpenVINO Core
    ov::Core core = ov::Core();

    // Initial 2 input Tensors
    ov::Tensor input1 = ov::Tensor(ov::element::f32, inpShape, vecConst[0]);
    ov::Tensor input2 = ov::Tensor(ov::element::f32, inpShape, vecConst[1]);
    initBuf(vecConst[0], shape_size(inpShape), 10.f); // std::numeric_limits<float>::min()
    initBuf(vecConst[1], shape_size(inpShape), 0.5f);

    // Construct model(subgraph)
    model = initModel(rtPrc, inpShape);

    // Compile model.
    auto compiledModel = core.compile_model(model, "CPU");
    auto inferRequest = compiledModel.create_infer_request();

    inferRequest.set_input_tensors(0, {input1});
    inferRequest.set_input_tensors(1, {input2});

    if (bSync)
    {
        inferRequest.infer();
    }
    else
    {
        inferRequest.start_async();
        inferRequest.wait();
    }

    outTensor = inferRequest.get_output_tensor();

    // Check output shape
    const auto &outShape = outTensor.get_shape();
    EXPECT_EQ(inpShape.size(), outShape.size());
    for (size_t i = 0; i < inpShape.size(); i++)
    {
        EXPECT_EQ(inpShape[i], outShape[i]);
    }
    // Check output data
    bool expectEQZero = false;
    const float *ouptputData = outTensor.data<float>();
    for (size_t i = 0; i < outTensor.get_size(); i++)
    {
        if (expectEQZero)
        {
            EXPECT_EQ(ouptputData[i], 0);
        }
        else
        {
            EXPECT_EQ(ouptputData[i], 5.f); // 5.87747e-39f
        }
    }

    for (size_t i = 0; i < vecConst.size(); i++)
    {
        if (vecConst[i])
            delete[] vecConst[i];
    }
    std::cout << "  Test model pass." << std::endl;
    return true;
}

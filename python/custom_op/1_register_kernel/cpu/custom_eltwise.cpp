// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_eltwise.hpp"

using namespace TemplateExtension;

MyEltwise::MyEltwise(const ov::OutputVector &args) : Op(args)
{
    constructor_validate_and_infer_types();
}

void MyEltwise::validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> MyEltwise::clone_with_new_inputs(const ov::OutputVector &new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 3, "Incorrect number of new arguments");
    return std::make_shared<MyEltwise>(new_args);
}

bool MyEltwise::visit_attributes(ov::AttributeVisitor &visitor)
{
    return true;
}

bool MyEltwise::evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const
{
    std::cout << "== MyEltwise::evaluate is called." << std::endl;
    float *inpData = reinterpret_cast<float *>(const_cast<void*>(inputs[0].data()));
    if (inputs[1].get_element_type() != ov::element::f32)
        OPENVINO_THROW("Unexpected bias type: " + inputs[1].get_element_type().to_string());
    float *pAlpha = reinterpret_cast<float *>(const_cast<void*>(inputs[1].data()));
    float *pBeta = reinterpret_cast<float *>(const_cast<void*>(inputs[2].data()));

    const auto &in = inputs[0];
    auto &out = outputs[0];

    out.set_shape(in.get_shape());
    auto total = in.get_size();
    if (in.get_element_type() == ov::element::f32)
    {
        auto *ptr_in = reinterpret_cast<float *>(in.data());
        auto *ptr_out = reinterpret_cast<float *>(out.data());
        for (size_t i = 0; i < total; i++)
        {
            ptr_out[i] = ptr_in[i] * pAlpha[0] + pBeta[0];
        }
    }
    else
    {
        std::cout << "Error: Not implemented for data type: " << in.get_element_type() << std::endl;
    }

    return true;
}

bool MyEltwise::has_evaluate() const
{
    return true;
}
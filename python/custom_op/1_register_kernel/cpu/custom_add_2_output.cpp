// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_add_2_output.hpp"

using namespace TemplateExtension;

MyAdd2Output::MyAdd2Output(const ov::OutputVector &args) : Op(args)
{
    constructor_validate_and_infer_types();
}

void MyAdd2Output::validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> MyAdd2Output::clone_with_new_inputs(const ov::OutputVector &new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 3, "Incorrect number of new arguments");
    return std::make_shared<MyAdd2Output>(new_args);
}

bool MyAdd2Output::visit_attributes(ov::AttributeVisitor &visitor)
{
    return true;
}

bool MyAdd2Output::evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const
{
    std::cout << "== MyAdd2Output::evaluate is called." << std::endl;
    float *inpData = reinterpret_cast<float *>(const_cast<void*>(inputs[0].data()));
    if (inputs[1].get_element_type() != ov::element::f32)
        OPENVINO_THROW("Unexpected bias type: " + inputs[1].get_element_type().to_string());
    float *pBias0 = reinterpret_cast<float *>(const_cast<void*>(inputs[1].data()));
    float *pBias1 = reinterpret_cast<float *>(const_cast<void*>(inputs[2].data()));

    const auto &in = inputs[0];
    auto &out0 = outputs[0];
    auto &out1 = outputs[1];

    out0.set_shape(in.get_shape());
    out1.set_shape(in.get_shape());

    auto total = in.get_size();
    if (in.get_element_type() == ov::element::f32)
    {
        auto *ptr_in = reinterpret_cast<float *>(in.data());
        auto *ptr_out1 = reinterpret_cast<float *>(out0.data());
        auto *ptr_out2 = reinterpret_cast<float *>(out1.data());
        for (size_t i = 0; i < total; i++)
        {
            ptr_out1[i] = ptr_in[i] + pBias0[0];
            ptr_out2[i] = ptr_in[i] + pBias1[0];
        }
    }
    else
    {
        std::cout << "Error: Not implemented for data type: " << in.get_element_type() << std::endl;
    }

    return true;
}

bool MyAdd2Output::has_evaluate() const
{
    return true;
}
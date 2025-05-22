#include <openvino.hpp>
#include <openvino/core/extension.hpp>
#include <openvino/op/op.hpp>

Identity::Identity(const ov::Output<ov::Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

void Identity::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> Identity::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<Identity>(new_args.at(0));
}

bool Identity::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

bool Identity::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& in = inputs[0];
    auto& out = outputs[0];
    if (out.data() == in.data())  // Nothing to do
        return true;
    out.set_shape(in.get_shape());
    memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool Identity::has_evaluate() const {
    return true;
}

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({

        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::Identity>>(),

        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::Identity>>()
    }));


#include <openvino/core/extension.hpp>
#include <openvino/op/op.hpp>

namespace TemplateExtension
{
    class MyAdd2Output : public ov::op::Op
    {
    public:
        // OP name.
        OPENVINO_OP("MyAdd2Output");

        MyAdd2Output() = default;
        MyAdd2Output(const ov::OutputVector &args); // const float &bias
        void validate_and_infer_types() override;
        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector &new_args) const override;
        bool visit_attributes(ov::AttributeVisitor &visitor) override;

        bool evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const override;
        bool has_evaluate() const override;

    private:
        // float _bias = 0;
    };
};
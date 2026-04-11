#include <iostream>
#include "private.hpp"

#define TEST(FUN) std::cout << "Start test: " << #FUN << std::endl; \
    bool ret = FUN; \
    std::cout << (ret ? "Done." : "Failure.") << std::endl

int main(int argc, char **argv)
{
    // TEST(test_model_multiply());
    // TEST(test_model_conv_bias_sum_reshape());
    // TEST(test_cpu_template_compare());
    // TEST(test_model_stateful());
    // TEST(test_model_if());
    // TEST(test_model_concat());
    // TEST(test_remote_tensor());
    // TEST(test_ov_model_with_custom_op());
    // TEST(test_merge_2_ov_ir());
    // TEST(test_build_position_ids());
    TEST(test_share_model_weights_for_2_same_model());
    return EXIT_SUCCESS;
}
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
    TEST(test_model_if());
    return EXIT_SUCCESS;
}
#pragma once

#include "utils.hpp"

bool test_model_multiply();
bool test_model_conv_bias_sum_reshape();
bool test_cpu_template_compare();
bool test_model_stateful();
bool test_model_if();
bool test_model_concat();
bool test_remote_tensor();
bool test_ov_model_with_custom_op();
bool test_merge_2_ov_ir();
bool test_build_position_ids();
bool test_share_model_weights_for_2_same_model();
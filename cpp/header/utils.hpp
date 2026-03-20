#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <openvino/openvino.hpp>

template <class T>
inline ov::Tensor initTensor(ov::element::Type type, const ov::Shape& shp, const std::vector<T>& inp_data)
{
    auto tensor = ov::Tensor(type, shp);
    auto size = tensor.get_size();
    assert(size == inp_data.size());
    std::memcpy(tensor.data<T>(), inp_data.data(), sizeof(T) * size);
    return tensor;
}

std::vector<float> randomData(const ov::Shape& shape);
std::vector<uint8_t> randomData_U8(const ov::Shape& shape);

#ifndef EXPECT_EQ
#define EXPECT_EQ(a, b)                                                                      \
    if ((a) != (b))                                                                          \
    {                                                                                        \
        std::cout << "Line:" << __LINE__ << ", Failed: " << #a << " != " << #b << std::endl; \
        std::cout << "  [" << a << " != " << b << "]" << std::endl;                          \
        exit(0);                                                                             \
    }
#endif
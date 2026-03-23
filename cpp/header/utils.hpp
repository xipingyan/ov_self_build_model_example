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
#include <openvino/opsets/opset1.hpp>

template <class T>
inline ov::Tensor initTensor(ov::element::Type type, const ov::Shape& shp, const std::vector<T>& inp_data)
{
    auto tensor = ov::Tensor(type, shp);
    auto size = tensor.get_size();
    OPENVINO_ASSERT(size == inp_data.size(), "Input shape: ", shp, " (size ", size, "), input data size: ", inp_data.size());
    std::memcpy(tensor.data<T>(), inp_data.data(), sizeof(T) * size);
    return tensor;
}

template <class T>
inline ov::Tensor initTensorWithValue0(ov::element::Type type, const ov::Shape& shp)
{
    auto tensor = ov::Tensor(type, shp);
    auto size = tensor.get_size();
    std::vector<T> inp_data(size, 0);
    std::memcpy(tensor.data<T>(), inp_data.data(), sizeof(T) * size);
    return tensor;
}

template <class T>
inline std::shared_ptr<ov::opset1::Constant> createConstant(const ov::element::Type& type, const ov::Shape& shape, const std::vector<T>& data) {
	auto tensor = initTensor(type, shape, data);
	return std::make_shared<ov::opset1::Constant>(tensor);
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
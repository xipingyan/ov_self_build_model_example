#pragma once

#include <openvino/openvino.hpp>

template <class T>
inline ov::Tensor initTensor(ov::element::Type type, ov::Shape shp, std::vector<T> inp_data)
{
    auto tensor = ov::Tensor(type, shp);
    memcpy(tensor.data<T>(), inp_data.data(), sizeof(T) * inp_data.size());
    return tensor;
}
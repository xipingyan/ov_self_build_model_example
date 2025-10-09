#pragma once

#include <random>
#include <assert.h>
#include <openvino/openvino.hpp>

template <class T>
inline ov::Tensor initTensor(ov::element::Type type, ov::Shape shp, std::vector<T> inp_data)
{
    auto tensor = ov::Tensor(type, shp);
    auto size = tensor.get_size();
    assert(size == inp_data.size());
    memcpy(tensor.data<T>(), inp_data.data(), sizeof(T) * size);
    return tensor;
}

inline std::vector<float> randomData(ov::Shape shape)
{
    size_t sz = std::accumulate(shape.begin(), shape.end(), 1, [](int a, int b)
                                { return a * b; });
    std::vector<float> rslt(sz);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (auto i = 0; i < sz; i++)
    {
        double random_number = distribution(generator);
        rslt[i] = random_number;
    }
    return rslt;
}

#include "utils.hpp"

#include <functional>

std::vector<float> randomData(const ov::Shape& shape)
{
	const size_t sz = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
									  std::multiplies<size_t>());
	std::vector<float> rslt(sz);

	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	for (size_t i = 0; i < sz; i++)
	{
		rslt[i] = distribution(generator);
	}
	return rslt;
}

std::vector<uint8_t> randomData_U8(const ov::Shape& shape)
{
	const size_t sz = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
									  std::multiplies<size_t>());
	std::vector<uint8_t> rslt(sz);

	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, 255);
	for (size_t i = 0; i < sz; i++)
	{
		rslt[i] = static_cast<uint8_t>(distribution(generator));
	}
	return rslt;
}

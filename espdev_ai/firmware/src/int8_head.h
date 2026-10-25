#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace pocketgpt_int8_head {

using Dot = int (*)(const std::int8_t*, const std::int8_t*, std::int32_t*, int);

inline bool quantize(const float* input, std::size_t count, std::size_t group,
                     std::int8_t* output, std::size_t output_capacity,
                     float* scales, std::size_t scale_capacity) {
    if (!input || !output || !scales || count == 0 || group == 0 ||
        group % 16 != 0 || count % group != 0 ||
        output_capacity < count || output_capacity - count < 16 ||
        scale_capacity < count / group) return false;
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(input[i])) return false;
    for (std::size_t base = 0; base < count; base += group) {
        float maximum = 0.0f;
        for (std::size_t i = 0; i < group; ++i)
            maximum = std::fmax(maximum, std::fabs(input[base + i]));
        float scale = maximum > 0.0f ? std::fmax(maximum / 127.0f, 1.0e-12f) : 1.0f;
        scales[base / group] = scale;
        for (std::size_t i = 0; i < group; ++i) {
            long value = std::lrintf(input[base + i] / scale);
            if (value > 127) value = 127;
            if (value < -127) value = -127;
            output[base + i] = static_cast<std::int8_t>(value);
        }
    }
    for (std::size_t i = count; i < count + 16; ++i) output[i] = 0;
    return true;
}

inline bool score_row(const std::int8_t* weights, const std::int8_t* input,
                      const float* input_scales, float weight_scale,
                      std::size_t count, std::size_t group, float* output, Dot dot) {
    if (!weights || !input || !input_scales || !output || !dot ||
        count == 0 || group == 0 || group % 16 != 0 || count % group != 0 ||
        !std::isfinite(weight_scale)) return false;
    float sum = 0.0f;
    for (std::size_t base = 0; base < count; base += group) {
        std::int32_t value = 0;
        if (dot(weights + base, input + base, &value, static_cast<int>(group)) != 0)
            return false;
        sum += static_cast<float>(value) * input_scales[base / group];
    }
    *output = sum * weight_scale;
    return std::isfinite(*output);
}

}

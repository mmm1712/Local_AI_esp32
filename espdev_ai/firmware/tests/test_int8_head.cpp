#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include "../src/int8_head.h"

int scalar_dot(const std::int8_t* a, const std::int8_t* b, std::int32_t* out, int n) {
    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += static_cast<int>(a[i]) * static_cast<int>(b[i]);
    assert(sum >= INT32_MIN && sum <= INT32_MAX);
    *out = static_cast<std::int32_t>(sum);
    return 0;
}

int failed_dot(const std::int8_t*, const std::int8_t*, std::int32_t*, int) { return -1; }

int main() {
    constexpr unsigned count = 448, group = 64;
    float x[count] = {}, scales[7] = {};
    std::int8_t q[count + 16], weights[count + 16];
    for (auto& value : q) value = 99;
    assert(pocketgpt_int8_head::quantize(x, count, group, q, count + 16, scales, 7));
    for (auto value : q) assert(value == 0);
    for (auto value : scales) assert(value == 1.0f);
    assert(!pocketgpt_int8_head::quantize(x, count, group, q, count, scales, 7));
    assert(!pocketgpt_int8_head::quantize(x, count, group, q, count + 16, scales, 6));
    assert(!pocketgpt_int8_head::quantize(x, count, 0, q, count + 16, scales, 7));
    assert(!pocketgpt_int8_head::quantize(x, count - 1, group, q, count + 16, scales, 7));
    assert(!pocketgpt_int8_head::quantize(nullptr, count, group, q, count + 16, scales, 7));
    x[100] = std::numeric_limits<float>::quiet_NaN();
    assert(!pocketgpt_int8_head::quantize(x, count, group, q, count + 16, scales, 7));
    x[100] = std::numeric_limits<float>::infinity();
    assert(!pocketgpt_int8_head::quantize(x, count, group, q, count + 16, scales, 7));
    for (unsigned i = 0; i < count; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.13f) * (1 + i / group);
        weights[i] = static_cast<std::int8_t>(static_cast<int>(i % 255) - 127);
    }
    assert(pocketgpt_int8_head::quantize(x, count, group, q, count + 16, scales, 7));
    float score = 0;
    assert(pocketgpt_int8_head::score_row(weights, q, scales, 0.013f, count, group, &score, scalar_dot));
    double reference = 0;
    for (unsigned i = 0; i < count; ++i)
        reference += static_cast<double>(weights[i]) * q[i] * scales[i / group] * 0.013f;
    assert(std::abs(score - reference) < 0.0001 * (1 + std::abs(reference)));
    for (unsigned i = count; i < count + 16; ++i) assert(q[i] == 0);
    assert(!pocketgpt_int8_head::score_row(weights, q, scales, 0.013f, count, group, &score, failed_dot));
    for (unsigned i = 0; i < count; ++i) { q[i] = 127; weights[i] = 127; }
    std::int32_t dot = 0;
    scalar_dot(weights, q, &dot, count);
    assert(dot == 7225792);
}

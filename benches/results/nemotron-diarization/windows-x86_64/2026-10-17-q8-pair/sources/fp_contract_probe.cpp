// Model-free compiler arithmetic observation; never part of a shipped runtime.
#if !defined(_WIN32) || !defined(_M_X64)
#error Actual Windows x64 arithmetic observation required
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

extern "C" __declspec(noinline) float contract_ordinary(float a, float b, float c) {
    return a * b + c;
}

extern "C" __declspec(noinline) float contract_separate(float a, float b, float c) {
    volatile float product = a * b;
    return product + c;
}

extern "C" __declspec(noinline) float contract_intrinsic(float a, float b, float c) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

static std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

int main() {
    // (1+2^-23)*(1-2^-23)-1 is -2^-46 when fused, zero with a rounded product.
    volatile std::uint32_t source[3] = {0x3f800001u, 0x3f7ffffeu, 0xbf800000u};
    float input[3];
    for (int i = 0; i < 3; ++i) {
        const std::uint32_t value = source[i];
        std::memcpy(&input[i], &value, sizeof(value));
    }
#ifdef __clang__
    const int clang_major = __clang_major__;
#else
    const int clang_major = 0;
#endif
#ifdef __FMA__
    const int fma_macro = 1;
#else
    const int fma_macro = 0;
#endif
#ifdef __AVX2__
    const int avx2_macro = 1;
#else
    const int avx2_macro = 0;
#endif
    const unsigned mxcsr = _mm_getcsr();
    const auto ordinary = bits(contract_ordinary(input[0], input[1], input[2]));
    const auto separate = bits(contract_separate(input[0], input[1], input[2]));
    const auto intrinsic = bits(contract_intrinsic(input[0], input[1], input[2]));
    std::printf("{\"inputs\":[%u,%u,%u],\"ordinary\":%u,\"separate\":%u,"
                "\"intrinsic\":%u,\"mxcsr\":%u,\"msc_ver\":%d,"
                "\"clang_major\":%d,\"fma_macro\":%d,\"avx2_macro\":%d}\n",
                bits(input[0]), bits(input[1]), bits(input[2]),
                ordinary, separate, intrinsic, mxcsr, _MSC_VER, clang_major,
                fma_macro, avx2_macro);
}

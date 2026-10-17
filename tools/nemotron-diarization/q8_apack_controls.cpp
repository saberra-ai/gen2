// Included by the checked benchmark derivation before any timed dispatch.
extern "C" void gen2_q8_apack_transport(const block_q8_0 *, int64_t, int64_t,
    int64_t, int64_t, void *, float *);
extern "C" size_t gen2_q8_apack_bound(int64_t, int64_t);

static void packing_controls(const std::string & prefix) {
    constexpr size_t count = 65536, k = count / 2, stride = k + 2;
    constexpr size_t maximum = (64u * 1024u * 1024u) / 36u;
    if (gen2_q8_apack_bound(0, 1) || gen2_q8_apack_bound(1, 0) ||
        gen2_q8_apack_bound(-1, 1) || gen2_q8_apack_bound(1, -1) ||
        gen2_q8_apack_bound(INT64_MAX, INT64_MAX) ||
        gen2_q8_apack_bound(1, maximum + 1) ||
        gen2_q8_apack_bound(1, maximum) != maximum ||
        gen2_q8_apack_bound(2, maximum / 2) != 2 * (maximum / 2) ||
        gen2_q8_apack_bound(2, maximum / 2 + 1) ||
        gen2_q8_apack_bound(2, k) != count)
        throw std::runtime_error("packing bounds failed");
    std::vector<block_q8_0> input(4 * stride);
    std::memset(input.data(), 0x5a, input.size() * sizeof(block_q8_0));
    for (size_t i = 0; i < count; ++i) {
        auto & block = input[(1 + i / k) * stride + i % k];
        block.d = static_cast<ggml_fp16_t>(i);
        auto * bytes = reinterpret_cast<unsigned char *>(block.qs);
        for (size_t j = 0; j < 32; ++j) bytes[j] = static_cast<unsigned char>((i + 17 * j) & 255);
    }
    const auto original = input;
    std::vector<unsigned char> payload(count * 32 + 64, 0xa5);
    std::vector<float> scales(count + 8, guard);
    gen2_q8_apack_transport(input.data(), stride, 1, 2, k, payload.data() + 32, scales.data() + 4);
    if (std::memcmp(input.data(), original.data(), input.size() * sizeof(block_q8_0)))
        throw std::runtime_error("packing changed source or padding");
    for (size_t j = 0; j < 32; ++j)
        if (payload[j] != 0xa5 || payload[count * 32 + 32 + j] != 0xa5)
            throw std::runtime_error("packing payload guard changed");
    for (size_t j = 0; j < 4; ++j)
        if (scales[j] != guard || scales[count + 4 + j] != guard)
            throw std::runtime_error("packing scale guard changed");
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(scales.data() + 4 + i, ggml_table_f32_f16 + i, sizeof(float)))
            throw std::runtime_error("packed scale differs from original lookup bits");
        for (size_t j = 0; j < 32; ++j)
            if (payload[32 + i * 32 + j] != static_cast<unsigned char>((i + 17 * j) & 255))
                throw std::runtime_error("packed payload differs");
    }
    save_output(prefix + ".apack-payload.i8", payload.data() + 32, count * 32);
    save_output(prefix + ".apack-scales.f32", scales.data() + 4, count * sizeof(float));
}

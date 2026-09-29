#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-hexagon/iq2-s-repack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ggml_hexagon_iq2s;

static int n_failed = 0;

static void check(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++n_failed;
    }
}

static uint32_t xorshift32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void test_sizes() {
    check(original_size_2d(256, 32) == 32u * sizeof(block_iq2_s),
          "original size for 32x256");
    check(repacked_size_2d(256, 32) == 8u * TILE_SIZE,
          "repacked size for 32x256");
    check(repacked_size_2d(256, 33) == 16u * TILE_SIZE,
          "row padding for 33x256");
    check(repacked_size_2d(512, 32) == 16u * TILE_SIZE,
          "two super-blocks per row");
    check(!valid_2d_shape(255, 32), "reject ne0 that is not a multiple of 256");
}

static void test_raw_roundtrip(int64_t ne0, int64_t ne1) {
    const size_t original_size = original_size_2d(ne0, ne1);
    const size_t repacked_size = repacked_size_2d(ne0, ne1);

    std::vector<uint8_t> original(original_size);
    std::vector<uint8_t> repacked(repacked_size);
    std::vector<uint8_t> roundtrip(original_size);

    uint32_t rng = 0x6d2b79f5u ^ (uint32_t) ne0 ^ ((uint32_t) ne1 << 16);
    for (uint8_t & v : original) {
        v = (uint8_t) xorshift32(rng);
    }

    check(repack_2d(
              reinterpret_cast<const block_iq2_s *>(original.data()), original.size(),
              ne0, ne1, repacked.data(), repacked.size()),
          "raw repack succeeds");

    check(unpack_2d(
              repacked.data(), repacked.size(), ne0, ne1,
              reinterpret_cast<block_iq2_s *>(roundtrip.data()), roundtrip.size()),
          "raw unpack succeeds");

    check(std::memcmp(original.data(), roundtrip.data(), original.size()) == 0,
          "raw IQ2_S bytes survive repack -> unpack exactly");
}

static void fill_source(std::vector<float> & src, int64_t ne0, int64_t ne1) {
    for (int64_t r = 0; r < ne1; ++r) {
        for (int64_t c = 0; c < ne0; ++c) {
            const float x = 0.37f * std::sin((float) (r * 17 + c) * 0.031f)
                          + 0.11f * std::cos((float) (r * 5  - c) * 0.017f)
                          + 0.003f * (float) ((c % 23) - 11);
            src[(size_t) r * (size_t) ne0 + (size_t) c] = x;
        }
    }
}

static void test_quantized_roundtrip(int64_t ne0, int64_t ne1) {
    const size_t original_size = original_size_2d(ne0, ne1);
    const size_t repacked_size = repacked_size_2d(ne0, ne1);

    std::vector<float> source((size_t) ne0 * (size_t) ne1);
    std::vector<float> imatrix(source.size(), 1.0f);
    std::vector<uint8_t> quantized(original_size);
    std::vector<uint8_t> repacked(repacked_size);
    std::vector<uint8_t> roundtrip(original_size);

    fill_source(source, ne0, ne1);

    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_IQ2_S,
        source.data(),
        quantized.data(),
        0,
        ne1,
        ne0,
        imatrix.data());

    check(written == quantized.size(), "ggml_quantize_chunk wrote expected IQ2_S byte count");

    check(repack_2d(
              reinterpret_cast<const block_iq2_s *>(quantized.data()), quantized.size(),
              ne0, ne1, repacked.data(), repacked.size()),
          "quantized repack succeeds");

    check(unpack_2d(
              repacked.data(), repacked.size(), ne0, ne1,
              reinterpret_cast<block_iq2_s *>(roundtrip.data()), roundtrip.size()),
          "quantized unpack succeeds");

    check(std::memcmp(quantized.data(), roundtrip.data(), quantized.size()) == 0,
          "valid IQ2_S bytes survive repack -> unpack exactly");

    const int64_t blocks_per_row = ne0 / QK_K;
    std::vector<float> deq_a((size_t) ne0);
    std::vector<float> deq_b((size_t) ne0);

    for (int64_t r = 0; r < ne1; ++r) {
        const block_iq2_s * a =
            reinterpret_cast<const block_iq2_s *>(quantized.data()) + r * blocks_per_row;
        const block_iq2_s * b =
            reinterpret_cast<const block_iq2_s *>(roundtrip.data()) + r * blocks_per_row;

        dequantize_row_iq2_s(a, deq_a.data(), ne0);
        dequantize_row_iq2_s(b, deq_b.data(), ne0);

        check(std::memcmp(deq_a.data(), deq_b.data(), (size_t) ne0 * sizeof(float)) == 0,
              "dequantized values are bit-identical after repack roundtrip");
    }
}

static void test_duplicate_d_validation() {
    const int64_t ne0 = 256;
    const int64_t ne1 = 1;

    std::vector<uint8_t> original(original_size_2d(ne0, ne1), 0);
    std::vector<uint8_t> repacked(repacked_size_2d(ne0, ne1), 0);
    std::vector<uint8_t> roundtrip(original.size(), 0);

    uint32_t rng = 0x12345678u;
    for (uint8_t & v : original) {
        v = (uint8_t) xorshift32(rng);
    }

    check(repack_2d(
              reinterpret_cast<const block_iq2_s *>(original.data()), original.size(),
              ne0, ne1, repacked.data(), repacked.size()),
          "corruption test repack succeeds");

    // Tile 1 belongs to the same 256-value super-block as tile 0, so its d
    // copy must match.  Flip one bit and ensure read-back detects it.
    repacked[TILE_SIZE + D_PLANE_OFFSET] ^= 0x01u;

    check(!unpack_2d(
              repacked.data(), repacked.size(), ne0, ne1,
              reinterpret_cast<block_iq2_s *>(roundtrip.data()), roundtrip.size()),
          "unpack rejects inconsistent duplicated d");
}

int main() {
    test_sizes();

    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1},
            {256, 31},
            {256, 32},
            {256, 33},
            {512, 32},
            {4096, 33},
        }) {
        test_raw_roundtrip(shape.first, shape.second);
    }

    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1},
            {256, 32},
            {512, 33},
            {4096, 32},
        }) {
        test_quantized_roundtrip(shape.first, shape.second);
    }

    test_duplicate_d_validation();

    ggml_quantize_free();

    if (n_failed == 0) {
        std::printf("PASS: IQ2_S phase-1 repack tests\n");
    } else {
        std::fprintf(stderr, "%d IQ2_S repack test(s) failed\n", n_failed);
    }

    return n_failed != 0;
}

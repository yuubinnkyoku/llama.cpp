#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "iq4-xs-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace iq4xs = ggml_hexagon_iq4xs;
static const int lut[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

struct test_case {
    const char * fixture;
    int64_t k, rows, cols, ne2, ne3;
    uint32_t seed;
    int hot = -1;
    int graph_mode = 0; // 1: shared activation pair, 2: MUL_MAT + ADD
};

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", message);
        std::exit(1);
    }
}

static uint32_t next_rand(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void set_scale(block_iq4_xs & b, int ib, int s) {
    unsigned ls = unsigned(s + 32);
    unsigned lo = 4 * (ib % 2), hi = 2 * ib;
    b.scales_l[ib / 2] = uint8_t((b.scales_l[ib / 2] & ~(15u << lo)) | ((ls & 15u) << lo));
    b.scales_h = uint16_t((b.scales_h & ~(3u << hi)) | ((ls >> 4) << hi));
}

static std::vector<block_iq4_xs> weights(const test_case & t) {
    const int scales[] = {-32, -31, -1, 0, 1, 30, 31, 7};
    const int alternating[] = {-32, 31, -16, 15, -1, 1, 0, 7};
    std::vector<block_iq4_xs> raw(size_t(t.k / 256 * t.rows * t.ne2 * t.ne3));
    uint32_t rng = t.seed;
    for (size_t i = 0; i < raw.size(); ++i) {
        auto & b = raw[i];
        bool random = std::strcmp(t.fixture, "random-safe") == 0;
        b.d = GGML_FP32_TO_FP16(random ? (0.01f + 0.002f * (next_rand(rng) % 100)) : (float(1 + i % 7) / 256));
        for (int ib = 0; ib < 8; ++ib) {
            int s = scales[ib];
            if (!std::strcmp(t.fixture, "one-hot")) s = scales[(ib + i) % 8];
            if (!std::strcmp(t.fixture, "zero")) s = 0;
            if (!std::strcmp(t.fixture, "alternating")) s = alternating[ib];
            if (random) s = int(next_rand(rng) % 64) - 32;
            set_scale(b, ib, s);
            for (int j = 0; j < 16; ++j) {
                unsigned lo = (j + ib + i) % 16;
                unsigned hi = (15 - j + 3 * ib + 5 * i) % 16;
                if (!std::strcmp(t.fixture, "alternating")) {
                    lo = j % 2 ? 15 - j / 2 : j / 2;
                    hi = lo;
                }
                if (random) { lo = next_rand(rng) % 16; hi = next_rand(rng) % 16; }
                b.qs[16 * ib + j] = uint8_t(lo | (hi << 4));
            }
        }
    }
    return raw;
}

static std::vector<float> activations(const test_case & t) {
    std::vector<float> x(size_t(t.k * t.cols * t.ne2 * t.ne3), 0);
    uint32_t rng = t.seed ^ 0xa5a5a5a5u;
    for (int64_t s = 0; s < t.ne2 * t.ne3; ++s) {
        for (int64_t c = 0; c < t.cols; ++c) {
            float * row = x.data() + (s * t.cols + c) * t.k;
            if (t.hot >= 0 || !std::strcmp(t.fixture, "lut")) {
                row[t.hot >= 0 ? t.hot : c % t.k] = (c % 2 ? -1.0f : 1.0f);
            } else {
                for (int64_t k = 0; k < t.k; ++k) {
                    if (!std::strcmp(t.fixture, "random-safe")) row[k] = float(int(next_rand(rng) % 2001) - 1000) / 1000;
                    else if (!std::strcmp(t.fixture, "alternating")) row[k] = k % 2 ? -1.0f : 1.0f;
                    else row[k] = next_rand(rng) % 2 ? -1.0f : 1.0f;
                }
            }
        }
    }
    return x;
}

static void validate_fixture(const test_case & t, const std::vector<block_iq4_xs> & raw) {
    for (const auto & b : raw) {
        float ref[256];
        dequantize_row_iq4_xs(&b, ref, 256);
        for (int ib = 0; ib < 8; ++ib) {
            require(iq4xs::iq4_xs_effective_scale_status(b, ib) == iq4xs::scale_status::OK, "safe scale preflight");
            uint8_t q[32];
            iq4xs::unpack_iq4_xs_indices_32(b, ib, q);
            for (int j = 0; j < 32; ++j) {
                require(ref[ib * 32 + j] == iq4xs::iq4_xs_effective_scale(b, ib) * lut[q[j]], "raw helper == CPU dequantizer");
            }
        }
    }
    iq4xs_alloc_state state;
    require(iq4xs_init_layout(state, t.k, t.rows, t.ne2, t.ne3), "preflight layout");
    std::vector<uint8_t> tiles(state.tile_slice_size * state.n_slices());
    require(iq4xs_prepare(state, (const uint8_t *) raw.data(), raw.size() * sizeof(raw[0]), tiles.data(), tiles.size()), "preflight prepare");
    require(state.state == GGML_HEXAGON_IQ4XS_REPACKED_SAFE, "preflight REPACKED_SAFE");
}

struct graph_run {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_tensor * w, * w2 = nullptr, * x, * bias = nullptr;
    std::vector<ggml_tensor *> outputs;
    ggml_gallocr_t alloc;

    graph_run(ggml_backend_t backend, const test_case & t) {
        ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
        ctx = ggml_init(params);
        require(ctx != nullptr, "context");
        w = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_XS, t.k, t.rows, t.ne2, t.ne3);
        x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, t.k, t.cols, t.ne2, t.ne3);
        ggml_set_name(w, "raw-iq4xs"); ggml_set_name(x, "shared-activation");
        graph = ggml_new_graph(ctx);
        auto y = ggml_mul_mat(ctx, w, x);
        ggml_set_name(y, "direct-iq4xs");
        if (t.graph_mode == 2) {
            bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, t.rows, t.cols, t.ne2, t.ne3);
            y = ggml_add(ctx, y, bias);
            ggml_set_name(y, "separate-add");
        }
        outputs.push_back(y);
        ggml_build_forward_expand(graph, y);
        if (t.graph_mode == 1) {
            w2 = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_XS, t.k, t.rows, t.ne2, t.ne3);
            ggml_set_name(w2, "raw-iq4xs-pair");
            auto y2 = ggml_mul_mat(ctx, w2, x);
            ggml_set_name(y2, "direct-iq4xs-pair");
            outputs.push_back(y2);
            ggml_build_forward_expand(graph, y2);
        }
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc, graph), "graph allocation");
    }
    ~graph_run() { ggml_gallocr_free(alloc); ggml_free(ctx); }

    void upload(const std::vector<block_iq4_xs> & raw, const std::vector<float> & act) {
        ggml_backend_tensor_set(w, raw.data(), 0, raw.size() * sizeof(raw[0]));
        if (w2) ggml_backend_tensor_set(w2, raw.data(), 0, raw.size() * sizeof(raw[0]));
        ggml_backend_tensor_set(x, act.data(), 0, act.size() * sizeof(float));
        if (bias) {
            std::vector<float> b(size_t(ggml_nelements(bias)), 0.25f);
            ggml_backend_tensor_set(bias, b.data(), 0, b.size() * sizeof(float));
        }
        std::vector<block_iq4_xs> back(raw.size());
        ggml_backend_tensor_get(w, back.data(), 0, back.size() * sizeof(back[0]));
        require(!std::memcmp(raw.data(), back.data(), back.size() * sizeof(back[0])), "exact raw readback");
        std::vector<float> xback(act.size());
        ggml_backend_tensor_get(x, xback.data(), 0, act.size() * sizeof(float));
        require(!std::memcmp(act.data(), xback.data(), act.size() * sizeof(float)), "exact activation readback");
    }
    std::vector<float> compute(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
        std::vector<float> out;
        for (auto y : outputs) {
            size_t offset = out.size();
            out.resize(offset + size_t(ggml_nelements(y)));
            ggml_backend_tensor_get(y, out.data() + offset, 0, ggml_nbytes(y));
        }
        return out;
    }
};

static bool compare(const test_case & t, const std::vector<block_iq4_xs> & raw, const std::vector<float> & act,
                    const std::vector<float> & cpu, const std::vector<float> & htp) {
    double max_abs = 0, max_rel = 0, squared = 0, ref_sq = 0, ref_abs = 0, product_l1 = 0, max_bound_ratio = 0;
    size_t worst = 0;
    bool ok = cpu.size() == htp.size();
    const size_t plane = size_t(t.rows * t.cols);
    const size_t tensor_size = plane * size_t(t.ne2 * t.ne3);
    std::vector<float> w(size_t(t.k));
    bool nonzero = false;
    for (size_t i = 0; i < cpu.size(); ++i) {
        size_t logical = i % tensor_size, row = logical % t.rows;
        size_t slice = logical / plane, col = logical / t.rows % t.cols;
        dequantize_row_iq4_xs(raw.data() + (slice * t.rows + row) * (t.k / 256), w.data(), t.k);
        const float * x = act.data() + (slice * t.cols + col) * t.k;
        double l1 = 0, peak_weighted = 0;
        for (int64_t sb = 0; sb < t.k; sb += 256) {
            double peak = 0;
            for (int j = 0; j < 256; ++j) peak = std::max(peak, std::fabs(double(x[sb + j])));
            for (int j = 0; j < 256; ++j) {
                l1 += std::fabs(double(w[sb + j]) * x[sb + j]);
                peak_weighted += std::fabs(double(w[sb + j])) * peak;
            }
        }
        double err = std::fabs(double(htp[i]) - cpu[i]);
        nonzero = nonzero || cpu[i] != 0;
        double bound = !std::strcmp(t.fixture, "random-safe") && t.hot < 0 ? 0.009 * peak_weighted + 0.001 * l1 + 1e-5 : 0.001 * l1 + 1e-5;
        if (err > max_abs) { max_abs = err; worst = i; }
        max_rel = std::max(max_rel, err / std::max(std::fabs(double(cpu[i])), 1e-6));
        max_bound_ratio = std::max(max_bound_ratio, err / bound);
        squared += err * err; ref_sq += double(cpu[i]) * cpu[i]; ref_abs += std::fabs(double(cpu[i])); product_l1 += l1;
        ok = ok && std::isfinite(cpu[i]) && std::isfinite(htp[i]) && err <= bound;
        if (!std::strcmp(t.fixture, "zero")) ok = ok && cpu[i] == 0 && htp[i] == 0;
        if ((row == 31 || row == 32 || row == 0 || row == size_t(t.rows - 1)) && col == 0 && i < tensor_size) {
            std::printf("BOUNDARY slice=%zu row=%zu cpu=%.9g htp=%.9g abs=%.9g\n", slice, row, cpu[i], htp[i], err);
        }
    }
    ok = ok && (nonzero || !std::strcmp(t.fixture, "zero"));
    double rms = std::sqrt(squared / cpu.size());
    std::printf("RESULT fixture=%s K=%lld rows=%lld cols=%lld ne2=%lld ne3=%lld seed=%u hot=%d graph=%d outputs=%zu max_abs=%.9g max_rel=%.9g RMS=%.9g normalized=%.9g NMSE=%.9g product_normalized=%.9g bound_ratio=%.9g worst_index=%zu status=%s\n",
                t.fixture, (long long)t.k, (long long)t.rows, (long long)t.cols, (long long)t.ne2, (long long)t.ne3,
                t.seed, t.hot, t.graph_mode, cpu.size(), max_abs, max_rel, rms, ref_abs ? rms / (ref_abs / cpu.size()) : 0,
                ref_sq ? squared / ref_sq : 0, product_l1 ? rms / (product_l1 / cpu.size()) : 0, max_bound_ratio, worst, ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<test_case> suite() {
    std::vector<test_case> cases;
    for (auto name : {"zero", "scale-boundary", "alternating"}) cases.push_back({name, 512, 33, 2, 1, 1, 0x13579bdfu});
    cases.push_back({"lut", 256, 33, 256, 1, 1, 0x13579bdfu});
    for (uint32_t seed : {1u, 7u, 42u, 0x13579bdfu, 0xdeadbeefu}) cases.push_back({"random-safe", 512, 33, 2, 1, 1, seed});
    for (int k : {256, 512}) for (int rows : {1, 31, 32, 33, 63, 64, 65}) cases.push_back({"scale-boundary", k, rows, 2, 1, 1, 42});
    for (int k : {256, 512, 768, 1024}) {
        std::vector<int> tested_hot;
        for (int hot : {0, 15, 16, 31, 32, 127, 128, 255, 256, k - 1}) {
            if (hot < k && std::find(tested_hot.begin(), tested_hot.end(), hot) == tested_hot.end()) {
                cases.push_back({"one-hot", k, 33, 1, 1, 1, 42, hot});
                tested_hot.push_back(hot);
            }
        }
    }
    for (int ne3 : {1, 2}) for (int rows : {32, 33}) for (int cols : {1, 2}) cases.push_back({"random-safe", 512, rows, cols, 2, ne3, 42});
    cases.push_back({"scale-boundary", 512, 33, 2, 1, 1, 42, -1, 1});
    cases.push_back({"scale-boundary", 512, 33, 2, 1, 1, 42, -1, 2});
    return cases;
}

static void host_test() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    require(cpu != nullptr, "CPU init");
    for (int s = -32; s <= 31; ++s) for (int ib = 0; ib < 8; ++ib) {
        block_iq4_xs b = {};
        set_scale(b, ib, s);
        require(iq4xs::iq4_xs_scale(b, ib) == s, "scale packing roundtrip");
        for (int other = 0; other < 8; ++other) if (other != ib) require(iq4xs::iq4_xs_scale(b, other) == -32, "scale field isolation");
    }
    for (const auto & t : suite()) {
        auto raw = weights(t);
        validate_fixture(t, raw);
        if (!std::strcmp(t.fixture, "lut")) {
            for (const auto & b : raw) for (int ib = 0; ib < 8; ++ib) {
                unsigned low = 0, high = 0;
                for (int j = 0; j < 16; ++j) {
                    low |= 1u << (b.qs[16 * ib + j] & 15);
                    high |= 1u << (b.qs[16 * ib + j] >> 4);
                }
                require(low == 0xffff && high == 0xffff, "LUT exhaustive in both nibble halves");
            }
        }
    }
    ggml_backend_free(cpu);
    std::printf("HOST fixture generator: PASS (all suite blocks; 512 scale packing/isolation cases)\n");
}

static void mutation(ggml_backend_t hex, ggml_backend_t cpu) {
    test_case t = {"scale-boundary", 512, 33, 2, 1, 1, 42};
    auto raw = weights(t); auto act = activations(t);
    validate_fixture(t, raw);
    graph_run h(hex, t), c(cpu, t);
    h.upload(raw, act); c.upload(raw, act);
    ggml_backend_t backends[] = {hex, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false, false);
    ggml_backend_sched_set_tensor_backend(sched, h.outputs[0], hex);
    require(ggml_backend_sched_alloc_graph(sched, h.graph), "public scheduler graph cache preparation");
    auto ref = c.compute(cpu);
    for (int i = 0; i < 2; ++i) {
        require(ggml_backend_sched_graph_compute(sched, h.graph) == GGML_STATUS_SUCCESS, "cached safe execution");
        std::vector<float> out(ref.size());
        ggml_backend_tensor_get(h.outputs[0], out.data(), 0, out.size() * sizeof(float));
        require(compare(t, raw, act, ref, out), "cached safe CPU comparison");
    }
    std::printf("MUTATION safe cache populated and reused\n");
    raw[3].d = GGML_FP32_TO_FP16(2048.0f); set_scale(raw[3], 0, -32);
    require(iq4xs::iq4_xs_effective_scale_status(raw[3], 0) == iq4xs::scale_status::TILE_SCALE_OVERFLOW, "overflow fixture");
    h.upload(raw, act); c.upload(raw, act);
    auto overflow_cpu = c.compute(cpu);
    for (float v : overflow_cpu) require(std::isfinite(v), "finite RAW CPU result");
    require(!ggml_backend_supports_op(hex, h.outputs[0]), "RAW_FALLBACK placement refusal");
    std::printf("MUTATION raw uploaded bit exact; CPU succeeds; attempting same cached graph\n");
    std::fflush(stdout);
    auto status = ggml_backend_sched_graph_compute_async(sched, h.graph);
    require(status != GGML_STATUS_SUCCESS, "cached RAW dispatch refused");
    std::printf("MUTATION PASS clean rejection status=%d\n", int(status));
    std::fflush(stdout);
    // The rejected session latches an error. End this isolated process without a failing synchronize.
    std::_Exit(0);
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 2 && !std::strcmp(argv[1], "--host")) { host_test(); return 0; }
    std::vector<test_case> cases = suite();
    if (argc >= 2 && !std::strcmp(argv[1], "--case")) {
        require(argc == 9 || argc == 10, "--case fixture K rows cols ne2 ne3 seed [hot]");
        test_case t = {argv[2], std::atoll(argv[3]), std::atoll(argv[4]), std::atoll(argv[5]), std::atoll(argv[6]), std::atoll(argv[7]), uint32_t(std::strtoul(argv[8], nullptr, 0))};
        if (argc == 10) t.hot = std::atoi(argv[9]);
        require(t.k > 0 && t.k % 256 == 0 && t.rows > 0 && t.cols > 0 && t.ne2 > 0 && t.ne3 > 0 && t.seed != 0 && t.hot < t.k, "valid exact shape/seed/hot");
        require(!std::strcmp(t.fixture, "zero") || !std::strcmp(t.fixture, "scale-boundary") || !std::strcmp(t.fixture, "lut") || !std::strcmp(t.fixture, "alternating") || !std::strcmp(t.fixture, "one-hot") || !std::strcmp(t.fixture, "random-safe"), "known fixture");
        cases = {t};
    } else require(argc == 1 || (argc == 2 && !std::strcmp(argv[1], "--mutation")), "usage: [--host | --mutation | --case fixture K rows cols ne2 ne3 seed [hot]]");
    auto cpu = ggml_backend_cpu_init();
    auto hex = ggml_backend_init_by_name("HTP0:0", nullptr);
    require(cpu && hex, "CPU and HTP0 init");
    ggml_backend_cpu_set_n_threads(cpu, 2);
    if (argc == 2 && !std::strcmp(argv[1], "--mutation")) mutation(hex, cpu);
    size_t passed = 0;
    for (const auto & t : cases) {
        std::printf("BEGIN fixture=%s K=%lld rows=%lld cols=%lld ne2=%lld ne3=%lld seed=%u hot=%d graph=%d\n", t.fixture, (long long)t.k, (long long)t.rows, (long long)t.cols, (long long)t.ne2, (long long)t.ne3, t.seed, t.hot, t.graph_mode);
        auto raw = weights(t); auto act = activations(t);
        validate_fixture(t, raw);
        graph_run c(cpu, t), h(hex, t);
        c.upload(raw, act); h.upload(raw, act);
        auto ref = c.compute(cpu); auto out = h.compute(hex);
        if (compare(t, raw, act, ref, out)) ++passed;
    }
    ggml_backend_free(hex); ggml_backend_free(cpu);
    std::printf("SUITE %zu/%zu %s\n", passed, cases.size(), passed == cases.size() ? "PASS" : "FAIL");
    return passed == cases.size() ? 0 : 1;
}

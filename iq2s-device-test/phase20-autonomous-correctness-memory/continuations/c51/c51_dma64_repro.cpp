// c51_dma64_repro - is the 0x2e trigger "an extended (DMA64) buffer used as an op output"?
//
//   c51_dma64_repro --out <wbuf|abuf> [--type T] [--m M] [--k K] [--repeats R]
//
// The 27B split never writes into a weights buffer: the host share and the DSP
// share both write their output into the activation buffer. The Phase 46 harness
// (c46_split_repeat) allocated w2, x2 and y2 from one context, so under DMA64=1
// the whole buffer became an extended weights mapping and the MUL_MAT output
// landed inside it. That is the one structural difference left between the two
// harnesses, so this test isolates it:
//
//   --out wbuf : the MUL_MAT result is allocated in the WEIGHTS buffer, so the
//                DSP writes its output into an extended mapping (c46's layout)
//   --out abuf : the MUL_MAT result is allocated in a separate default buffer,
//                so the extended mapping is read only (the split's layout)
//
// Everything else (type, shapes, input data, op) is identical, so a difference
// in rc between the two runs is the trigger.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_type type_by_name(const char * name) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && strcmp(n, name) == 0) {
            return (ggml_type) i;
        }
    }
    return GGML_TYPE_COUNT;
}

static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// deterministic host image in the original GGUF row layout
static std::vector<uint8_t> host_image(ggml_type at, int k, int rows, uint32_t seed) {
    const size_t nels = (size_t) k * rows;
    std::vector<float> vals(nels);
    uint32_t x = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < nels; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        vals[i] = (float) ((int32_t) (x % 2001) - 1000) / 1000.0f;
    }
    std::vector<uint8_t> buf(ggml_row_size(at, k) * rows, 0);
    if (at == GGML_TYPE_F32) {
        memcpy(buf.data(), vals.data(), nels * sizeof(float));
    } else if (at == GGML_TYPE_F16) {
        ggml_fp16_t * p = (ggml_fp16_t *) buf.data();
        for (size_t i = 0; i < nels; i++) {
            p[i] = ggml_fp32_to_fp16(vals[i]);
        }
    } else {
        const size_t bs = ggml_blck_size(at);
        ggml_quantize_chunk(at, vals.data(), buf.data(), 0, nels / bs, bs, nullptr);
    }
    return buf;
}

int main(int argc, char ** argv) {
    std::string out = "wbuf";
    std::string tname = "iq2_s";
    int m = 17408;
    int k = 5120;
    int repeats = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--out")     { out = next(); }
        else if (a == "--type")    { tname = next(); }
        else if (a == "--m")       { m = atoi(next()); }
        else if (a == "--k")       { k = atoi(next()); }
        else if (a == "--repeats") { repeats = atoi(next()); }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    const bool out_in_wbuf = (out == "wbuf");
    if (!out_in_wbuf && out != "abuf") {
        fprintf(stderr, "--out must be wbuf or abuf\n");
        return 2;
    }
    const ggml_type at = type_by_name(tname.c_str());
    if (at == GGML_TYPE_COUNT || m < 2 || k < 1 || repeats < 1) {
        return 2;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
        if (r && strcmp(ggml_backend_reg_name(r), "HTP") == 0) {
            dev = d;
            break;
        }
    }
    if (!dev) { fprintf(stderr, "no HTP device\n"); return 3; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "cannot init the HTP backend\n"); return 3; }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t row_bytes = ggml_row_size(at, k);
    std::vector<uint8_t> wimg = host_image(at, k, m, 11);
    std::vector<float>   act((size_t) k);
    for (int i = 0; i < k; i++) {
        act[i] = 0.001f * (float) ((i * 37) % 101) - 0.05f;
    }

    printf("out %s type %s m %d k %d repeats %d\n", out.c_str(), ggml_type_name(at), m, k, repeats);
    fflush(stdout);

    // The weights context. Under --out wbuf the MUL_MAT result joins it, so the
    // whole buffer is a WEIGHTS buffer and the DSP writes its result into an
    // extended mapping, exactly as c46_split_repeat's b2 did.
    ggml_init_params wp = { 256u * 1024u * 1024u, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, at, k, m);

    // The activation context: the input, and the result unless it went to wbuf.
    ggml_init_params ap = { 256u * 1024u * 1024u, nullptr, true };
    ggml_context * actx = ggml_init(ap);
    ggml_tensor * x = ggml_new_tensor_2d(actx, GGML_TYPE_F32, k, 1);

    // The op lives in the context that owns its result, so no tensor is moved
    // between buffers after allocation.
    ggml_context * yctx = out_in_wbuf ? wctx : actx;
    ggml_tensor * y = ggml_mul_mat(yctx, w, x);
    ggml_set_name(y, "dma64-repro");
    if (!ggml_backend_supports_op(be, y)) {
        fprintf(stderr, "op not supported\n");
        return 6;
    }

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) { fprintf(stderr, "cannot allocate the weights\n"); return 3; }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors_from_buft(actx, buft);
    if (!abuf) { fprintf(stderr, "cannot allocate the activations\n"); return 3; }

    ggml_backend_tensor_set(w, wimg.data(), 0, row_bytes * (size_t) m);
    ggml_backend_tensor_set(x, act.data(), 0, act.size() * sizeof(float));

    printf("wbuf size %zu usage %d, abuf size %zu usage %d\n",
            (size_t) ggml_backend_buffer_get_size(wbuf), (int) ggml_backend_buffer_get_usage(wbuf),
            (size_t) ggml_backend_buffer_get_size(abuf), (int) ggml_backend_buffer_get_usage(abuf));
    printf("y in %s\n", out_in_wbuf ? "wbuf" : "abuf");
    fflush(stdout);

    ggml_cgraph * gf = ggml_new_graph(yctx);
    ggml_build_forward_expand(gf, y);

    for (int rep = 0; rep < repeats; rep++) {
        const ggml_status st = ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        std::vector<float> got((size_t) m);
        ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
        size_t nonfinite = 0;
        for (float v : got) {
            if (!std::isfinite(v)) nonfinite++;
        }
        printf("REPEAT %d status %d hash %016llx nonfinite %zu f0 %.6f f1 %.6f\n", rep,
                (int) st, (unsigned long long) fnv1a(got.data(), got.size() * sizeof(float)),
                nonfinite, (double) got[0], (double) got[1]);
        fflush(stdout);
    }

    ggml_backend_buffer_free(abuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(actx);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}

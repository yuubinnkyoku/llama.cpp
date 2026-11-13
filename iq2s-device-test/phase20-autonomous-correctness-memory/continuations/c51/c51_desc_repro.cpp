// c51_desc_repro - does a cached opbatch descriptor follow a shape change?
//
//   c51_desc_repro [--repeats T]
//
// The opbatch caches one htp_tensor descriptor per ggml_tensor pointer. A tensor
// can be reused with a different shape between two graph computes that land in
// the same batch (no flush in between). The old code returned the cached
// descriptor for the pointer without checking the shape, so the DSP read the
// previous size. This test changes ne[1]/nb[1] of the same weight tensor object
// between two computes and checks that both computes still produce the right
// answer.
//
// The weight is F32 so the arithmetic is exact: a wrong descriptor size shows up
// as a wrong number, not as a rounding difference.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static const int64_t K  = 256;
static const int64_t M0 = 8;
static const int64_t M1 = 16;

static void fill_w(std::vector<float> & wv) {
    for (int64_t r = 0; r < M1; r++) {
        for (int64_t c = 0; c < K; c++) {
            wv[(size_t) r * K + c] = (float) ((r * 7 + c * 3) % 17) * 0.125f;
        }
    }
}

static void fill_x(std::vector<float> & xv) {
    for (int64_t c = 0; c < K; c++) {
        xv[(size_t) c] = (float) ((c * 5) % 11) * 0.25f;
    }
}

static float ref_cell(const std::vector<float> & wv, const std::vector<float> & xv, int64_t r) {
    double acc = 0.0;
    for (int64_t c = 0; c < K; c++) { acc += (double) wv[(size_t) r * K + c] * (double) xv[(size_t) c]; }
    return (float) acc;
}

int main(int argc, char ** argv) {
    int repeats = 3;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--repeats" && i + 1 < argc) { repeats = atoi(argv[++i]); }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    setenv("GGML_HEXAGON_DEVICES", "HTP0:0", 1);
    setenv("GGML_HEXAGON_OPPOLL",  "0", 1);
    setenv("GGML_HEXAGON_DMA64",   "0", 1);
    setenv("GGML_HEXAGON_SPLIT",   "0", 1);

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
        if (r && strcmp(ggml_backend_reg_name(r), "HTP") == 0) { dev = d; break; }
    }
    if (!dev) { fprintf(stderr, "no HTP device\n"); return 3; }
    printf("device %s\n", ggml_backend_dev_name(dev));

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_t             be   = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "cannot init the HTP backend\n"); return 3; }

    std::vector<float> wv((size_t) K * M1), xv((size_t) K);
    fill_w(wv);
    fill_x(xv);

    // Persistent weight and activation. The weight is allocated at the larger
    // shape and only its ne[1]/nb[1] change, so no allocation is resized.
    ggml_init_params wp = { 4u * 1024u * 1024u, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M1);
    ggml_set_name(w, "w");
    ggml_tensor * x = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, 1);
    ggml_set_name(x, "x");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) { fprintf(stderr, "cannot allocate the weights\n"); return 3; }
    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));

    int bad = 0;
    for (int rep = 0; rep < repeats; rep++) {
        // Two computes in a row with NO synchronize between them, so both land in
        // the same opbatch. The weight object is the same; only its shape moves.
        // The outputs are read after one join at the end, so the batch really is
        // shared and the descriptor cache is exercised.
        float out0[M0], out1[M1];
        ggml_tensor * y0 = nullptr;
        ggml_tensor * y1 = nullptr;
        ggml_context * actx = nullptr;
        ggml_backend_buffer_t abuf = nullptr;

        for (int step = 0; step < 2; step++) {
            const int64_t m = step == 0 ? M0 : M1;
            w->ne[1] = m;
            w->nb[1] = (size_t) K * sizeof(float);

            ggml_init_params ap = { 8u * 1024u * 1024u, nullptr, true };
            actx = ggml_init(ap);
            ggml_tensor * y = ggml_mul_mat(actx, w, x);
            ggml_set_name(y, step == 0 ? "y0" : "y1");
            abuf = ggml_backend_alloc_ctx_tensors_from_buft(actx, buft);
            if (!abuf) { fprintf(stderr, "cannot allocate the output\n"); return 3; }

            ggml_cgraph * g = ggml_new_graph_custom(actx, 64, false);
            ggml_build_forward_expand(g, y);
            // async: do NOT join here, so both computes share one opbatch and the
            // descriptor cache is exercised across the shape change
            ggml_backend_graph_compute_async(be, g);
            if (step == 0) { y0 = y; } else { y1 = y; }
        }

        // one join for both computes
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(y0, out0, 0, sizeof(out0));
        ggml_backend_tensor_get(y1, out1, 0, sizeof(out1));
        ggml_backend_buffer_free(abuf);
        ggml_free(actx);

        int d0 = 0, d1 = 0;
        for (int64_t r = 0; r < M0; r++) { if (out0[r] != ref_cell(wv, xv, r)) d0++; }
        for (int64_t r = 0; r < M1; r++) { if (out1[r] != ref_cell(wv, xv, r)) d1++; }
        printf("REPEAT %d shape0_diff %d shape1_diff %d hash0 %016llx hash1 %016llx\n",
                rep, d0, d1,
                (unsigned long long) fnv1a(out0, sizeof(out0)),
                (unsigned long long) fnv1a(out1, sizeof(out1)));
        if (d0 || d1) bad++;
    }

    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    printf("DESC_REPRO %s bad_repeats %d\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}

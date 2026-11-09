// Phase 49.5 - measured CPU side matvec bandwidth per weight type.
//
//   p495_cpu_matvec_test [K] [N] [iters]
//
// The 27B decode on the CPU path reads about 8 GiB of weights per token, so the
// CPU split time is set by the effective weight read bandwidth. This measures
// that bandwidth on the CPU backend for every type the model uses, at the shape
// of a single token step (M=1).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed) {}
    float uni() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float) ((s >> 33) & 0xFFFFFF) / 16777216.0f - 0.5f;
    }
};

static int bench(ggml_backend_t be, ggml_backend_buffer_type_t bt, enum ggml_type wt, int K, int N, int iters) {
    const int M = 1;

    ggml_init_params ip = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        printf("P495MV FAIL ctx\n");
        return 1;
    }

    ggml_tensor * tw = ggml_new_tensor_2d(ctx, wt, K, N);
    ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * ty = ggml_mul_mat(ctx, tw, tx);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ty);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt);
    if (buf == nullptr) {
        printf("P495MV FAIL buffer %s\n", ggml_type_name(wt));
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> xv((size_t) K * M, 0.5f);
    ggml_backend_tensor_set(tx, xv.data(), 0, ggml_nbytes(tx));

    if (wt == GGML_TYPE_BF16) {
        std::vector<uint16_t> wb((size_t) K * N, f2bf(0.5f));
        ggml_backend_tensor_set(tw, wb.data(), 0, ggml_nbytes(tw));
    } else if (wt == GGML_TYPE_F32) {
        std::vector<float> wf((size_t) K * N, 0.5f);
        ggml_backend_tensor_set(tw, wf.data(), 0, ggml_nbytes(tw));
    } else {
        rng r(0x51ed);
        std::vector<float> wf((size_t) K * N);
        for (size_t i = 0; i < wf.size(); i++) {
            wf[i] = r.uni();
        }
        std::vector<uint8_t> wq(ggml_nbytes(tw));
        // IQ2_XXS refuses to quantize without an importance matrix, a flat one is fine here
        std::vector<float> im((size_t) K, 1.0f);
        const size_t written = ggml_quantize_chunk(wt, wf.data(), wq.data(), 0, N, K, im.data());
        if (written != wq.size()) {
            printf("P495MV FAIL quantize %s wrote %zu of %zu\n", ggml_type_name(wt), written, wq.size());
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return 1;
        }
        ggml_backend_tensor_set(tw, wq.data(), 0, ggml_nbytes(tw));
    }

    float y[8] = { 0 };
    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute(be, gf);
    }
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < iters; i++) {
        ggml_backend_graph_compute(be, gf);
    }
    const int64_t t1 = ggml_time_us();
    ggml_backend_tensor_get(ty, y, 0, 4);

    const double secs = (double) (t1 - t0) / 1e6;
    const double gbs  = (double) ggml_nbytes(tw) * iters / secs / 1e9;
    printf("P495MV type %-7s K %5d N %6d M %d iters %3d per_op_us %9.1f gbs %6.2f y0 %.6g\n",
           ggml_type_name(wt), K, N, M, iters, (double) (t1 - t0) / iters, gbs, (double) y[0]);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    const int K     = argc > 1 ? atoi(argv[1]) : 5120;
    const int N     = argc > 2 ? atoi(argv[2]) : 4096;
    const int iters = argc > 3 ? atoi(argv[3]) : 20;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        printf("P495MV FAIL no cpu device\n");
        return 1;
    }
    ggml_backend_t             be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t bt = ggml_backend_dev_buffer_type(dev);
    if (be == nullptr || bt == nullptr) {
        printf("P495MV FAIL cpu init\n");
        return 1;
    }

    const int n_threads = argc > 4 ? atoi(argv[4]) : 7;
    ggml_backend_cpu_set_n_threads(be, n_threads);

    printf("P495MV cpu %s K %d N %d iters %d threads %d\n", ggml_backend_name(be), K, N, iters, n_threads);
    const enum ggml_type types[] = {
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
        GGML_TYPE_Q4_K, GGML_TYPE_BF16, GGML_TYPE_F32,
    };

    int rc = 0;
    for (enum ggml_type t : types) {
        const int it = (t == GGML_TYPE_F32 || t == GGML_TYPE_BF16) ? (iters / 4 > 4 ? iters / 4 : 4) : iters;
        rc |= bench(be, bt, t, K, N, it);
    }

    ggml_backend_free(be);
    printf("P495MV done rc %d\n", rc);
    return rc;
}

// c51_split_repro - run the row split over the 34 weights, without the 27B model.
//
//   c51_split_repro --mode <off|seq|async> [--n N] [--repeats T] [--rows R]
//                   [--frac F] [--dump P] [--small]
//
// The weights are built in an HTP weights buffer with the names and shapes the
// model uses, so GGML_HEXAGON_SPLIT_WEIGHTS matches the real list and the split
// takes the same path. The graph is one column MUL_MAT per weight, in device
// memory, exactly like a decode step. Every repeat prints a hash of all outputs,
// so a repeat that differs is a reproduction, and the dump mode leaves the per
// op output and the full DSP reference for p51_dump_cmp.py.
//
// This is the small reproduction: no model, no 9 GB read, runs in seconds.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct wdesc {
    const char * name;
    ggml_type    type;
    int64_t      k;
    int64_t      m;
};

// the 34 split weights of layers 56-63, as the model split log prints them
static const wdesc k_weights[] = {
    { "blk.56.attn_qkv.weight",   GGML_TYPE_IQ3_XXS,  5120, 10240 },
    { "blk.56.attn_gate.weight",  GGML_TYPE_IQ3_XXS,  5120,  6144 },
    { "blk.56.ssm_out.weight",    GGML_TYPE_IQ3_S,    6144,  5120 },
    { "blk.56.ffn_gate.weight",   GGML_TYPE_IQ3_S,    5120, 17408 },
    { "blk.56.ffn_up.weight",     GGML_TYPE_IQ3_XXS,  5120, 17408 },
    { "blk.57.attn_qkv.weight",   GGML_TYPE_IQ3_XXS,  5120, 10240 },
    { "blk.57.attn_gate.weight",  GGML_TYPE_IQ3_XXS,  5120,  6144 },
    { "blk.57.ssm_out.weight",    GGML_TYPE_IQ3_S,    6144,  5120 },
    { "blk.57.ffn_gate.weight",   GGML_TYPE_IQ3_XXS,  5120, 17408 },
    { "blk.57.ffn_down.weight",   GGML_TYPE_IQ3_S,   17408,  5120 },
    { "blk.57.ffn_up.weight",     GGML_TYPE_IQ3_S,    5120, 17408 },
    { "blk.58.attn_qkv.weight",   GGML_TYPE_IQ3_XXS,  5120, 10240 },
    { "blk.58.attn_gate.weight",  GGML_TYPE_IQ3_XXS,  5120,  6144 },
    { "blk.58.ssm_out.weight",    GGML_TYPE_IQ3_S,    6144,  5120 },
    { "blk.58.ffn_gate.weight",   GGML_TYPE_IQ3_XXS,  5120, 17408 },
    { "blk.58.ffn_down.weight",   GGML_TYPE_IQ3_S,   17408,  5120 },
    { "blk.58.ffn_up.weight",     GGML_TYPE_IQ3_S,    5120, 17408 },
    { "blk.59.attn_output.weight", GGML_TYPE_IQ3_S,   6144,  5120 },
    { "blk.59.ffn_gate.weight",   GGML_TYPE_IQ3_S,    5120, 17408 },
    { "blk.59.ffn_down.weight",   GGML_TYPE_IQ3_XXS, 17408,  5120 },
    { "blk.60.attn_qkv.weight",   GGML_TYPE_IQ3_XXS,  5120, 10240 },
    { "blk.60.ssm_out.weight",    GGML_TYPE_IQ3_XXS,  6144,  5120 },
    { "blk.60.ffn_gate.weight",   GGML_TYPE_IQ3_XXS,  5120, 17408 },
    { "blk.60.ffn_down.weight",   GGML_TYPE_IQ3_XXS, 17408,  5120 },
    { "blk.60.ffn_up.weight",     GGML_TYPE_IQ3_S,    5120, 17408 },
    { "blk.61.ssm_out.weight",    GGML_TYPE_IQ3_S,    6144,  5120 },
    { "blk.61.ffn_gate.weight",   GGML_TYPE_IQ3_XXS,  5120, 17408 },
    { "blk.61.ffn_down.weight",   GGML_TYPE_IQ3_XXS, 17408,  5120 },
    { "blk.62.ssm_out.weight",    GGML_TYPE_IQ3_S,    6144,  5120 },
    { "blk.62.ffn_down.weight",   GGML_TYPE_IQ3_S,   17408,  5120 },
    { "blk.62.ffn_up.weight",     GGML_TYPE_IQ3_S,    5120, 17408 },
    { "blk.63.attn_k.weight",     GGML_TYPE_IQ3_XXS,  5120,  1024 },
    { "blk.63.attn_v.weight",     GGML_TYPE_IQ3_S,    5120,  1024 },
    { "blk.63.attn_output.weight", GGML_TYPE_IQ3_S,   6144,  5120 },
};
static const int k_nall = (int) (sizeof(k_weights) / sizeof(k_weights[0]));

static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void setenv_or_die(const char * k, const std::string & v) {
    if (setenv(k, v.c_str(), 1) != 0) {
        fprintf(stderr, "setenv %s failed\n", k);
        exit(1);
    }
}

int main(int argc, char ** argv) {
    std::string mode   = "async";
    int         n      = k_nall;
    int         repeats = 3;
    int         rows   = 0;          // 0 = use --frac
    double      frac   = 0.25;
    std::string dump;
    bool        small  = false;
    bool        chain  = true;
    bool        replicate = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--mode")    { mode = next(); }
        else if (a == "--n")       { n = atoi(next()); }
        else if (a == "--repeats") { repeats = atoi(next()); }
        else if (a == "--rows")    { rows = atoi(next()); }
        else if (a == "--frac")    { frac = atof(next()); }
        else if (a == "--dump")    { dump = next(); }
        else if (a == "--small")   { small = true; }
        else if (a == "--nochain") { chain = false; }
        else if (a == "--replicate") { replicate = true; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (n < 1 || n > k_nall) { fprintf(stderr, "--n out of range\n"); return 2; }

    // the split reads these once, so they have to be set before the backend starts
    std::string names;
    for (int i = 0; i < n; i++) {
        if (i) names += ",";
        names += k_weights[i].name;
    }
    setenv_or_die("GGML_HEXAGON_SPLIT",         mode == "off" ? "0" : "1");
    setenv_or_die("GGML_HEXAGON_SPLIT_SEQ",     mode == "seq" ? "1" : "0");
    setenv_or_die("GGML_HEXAGON_SPLIT_WEIGHTS", names);
    setenv_or_die("GGML_HEXAGON_SPLIT_THREADS", "6");
    if (rows > 0) {
        setenv_or_die("GGML_HEXAGON_SPLIT_CPU_ROWS", std::to_string(rows));
    } else {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.4f", frac);
        setenv_or_die("GGML_HEXAGON_SPLIT_CPU_FRAC", buf);
    }
    setenv_or_die("GGML_HEXAGON_DEVICES", "HTP0:0");
    setenv_or_die("GGML_HEXAGON_OPPOLL",  "0");
    setenv_or_die("GGML_HEXAGON_DMA64",   "0");
    setenv_or_die("GGML_HEXAGON_IQ2S_GATHER",  "1");
    setenv_or_die("GGML_HEXAGON_IQ3S_GATHER",  "1");
    setenv_or_die("GGML_HEXAGON_IQ3XXS_GATHER","1");
    setenv_or_die("GGML_HEXAGON_FFN_OFFLOAD",  "1");
    if (!dump.empty()) {
        setenv_or_die("GGML_HEXAGON_SPLIT_DUMP",     dump);
        setenv_or_die("GGML_HEXAGON_SPLIT_DUMP_REF", dump + ".ref");
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
    printf("device %s : %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    printf("mode %s n %d repeats %d rows %d frac %.4f chain %d dump %s\n", mode.c_str(), n,
            repeats, rows, frac, (int) chain, dump.empty() ? "none" : dump.c_str());
    printf("replicate %d\n", (int) replicate);

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_t             be   = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "cannot init the HTP backend\n"); return 3; }

    // weights
    ggml_init_params wp = { 8u * 1024u * 1024u, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    std::vector<ggml_tensor *> w(n);
    for (int i = 0; i < n; i++) {
        int64_t k = k_weights[i].k;
        int64_t m = small ? 512 : k_weights[i].m;
        w[i] = ggml_new_tensor_2d(wctx, k_weights[i].type, k, m);
        ggml_set_name(w[i], k_weights[i].name);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) { fprintf(stderr, "cannot allocate the weights\n"); return 3; }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    size_t wbytes = 0;
        for (int i = 0; i < n; i++) {
        ggml_tensor * t  = w[i];
        const int64_t k  = t->ne[0];
        const int64_t m  = t->ne[1];
        const size_t  rb = ggml_row_size(t->type, k);
        std::vector<float> row(k);
        std::vector<uint8_t> dst(rb * m);
        // --replicate quantizes one row and copies it to the rest. The real shapes
        // then take seconds instead of minutes, at the cost of not noticing a row
        // index shift; use it for the memory and boundary path only.
        for (int64_t r = 0; r < m; r++) {
            if (replicate && r > 0) {
                memcpy(dst.data() + r * rb, dst.data(), rb);
                continue;
            }
            for (int64_t c = 0; c < k; c++) {
                uint32_t u = (uint32_t) (r * 131 + c * 7);
                row[c] = ((u * 2654435761u) >> 9 & 0xFFFF) / 32768.0f - 1.0f;
            }
            size_t got = ggml_quantize_chunk(t->type, row.data(), dst.data() + r * rb, 0, 1, k, nullptr);
            if (got != rb) { fprintf(stderr, "%s: row %zu != %zu\n", t->name, got, rb); return 4; }
        }
        ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
        wbytes += dst.size();
    }
    printf("weights %d, %.2f MiB set\n", n, wbytes / (1024.0 * 1024.0));

    // activations and outputs, in device memory so the split can write the tail.
    // The default chain puts an ADD in front of every MUL_MAT, so the split input
    // is written by a DSP op of the same batch, as it is in the model. Without it
    // the input is a graph input and no producer can be in flight.
    ggml_init_params ap = { 8u * 1024u * 1024u, nullptr, true };
    ggml_context * actx = ggml_init(ap);
    std::vector<ggml_tensor *> x(n), xs(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = ggml_new_tensor_2d(actx, GGML_TYPE_F32, w[i]->ne[0], 1);
        ggml_set_name(x[i], (std::string("x_") + k_weights[i].name).c_str());
        ggml_tensor * src = x[i];
        if (chain) {
            xs[i] = ggml_scale(actx, x[i], 2.0f);
            ggml_set_name(xs[i], (std::string("xs_") + k_weights[i].name).c_str());
            src = xs[i];
        }
        y[i] = ggml_mul_mat(actx, w[i], src);
        ggml_set_name(y[i], (std::string("y_") + k_weights[i].name).c_str());
    }
    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors_from_buft(actx, buft);
    if (!abuf) { fprintf(stderr, "cannot allocate the activations\n"); return 3; }

    for (int i = 0; i < n; i++) {
        std::vector<float> xv(x[i]->ne[0]);
        for (size_t c = 0; c < xv.size(); c++) {
            uint32_t u = (uint32_t) (c * 2654435761u);
            xv[c] = ((u >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        }
        ggml_backend_tensor_set(x[i], xv.data(), 0, xv.size() * sizeof(float));
    }

    ggml_cgraph * gf = ggml_new_graph_custom(actx, 512, false);
    for (int i = 0; i < n; i++) {
        ggml_build_forward_expand(gf, y[i]);
    }
    printf("graph nodes %d\n", ggml_graph_n_nodes(gf));

    for (int rep = 0; rep < repeats; rep++) {
        ggml_status st = ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        uint64_t h = 1469598103934665603ull;
        size_t   nonfinite = 0;
        for (int i = 0; i < n; i++) {
            std::vector<float> yv(y[i]->ne[0]);
            ggml_backend_tensor_get(y[i], yv.data(), 0, yv.size() * sizeof(float));
            for (float v : yv) {
                if (!std::isfinite(v)) nonfinite++;
            }
            h = fnv1a(yv.data(), yv.size() * sizeof(float), h);
        }
        printf("REPEAT %d status %d hash %016llx nonfinite %zu\n", rep, (int) st,
                (unsigned long long) h, nonfinite);
    }

    ggml_backend_buffer_free(abuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(actx);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}

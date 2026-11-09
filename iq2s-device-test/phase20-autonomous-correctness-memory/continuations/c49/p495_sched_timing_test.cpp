// Phase 49.5 - independent checks of the scheduler split timing.
//
//   p495_sched_timing_test mix   [graphs] [every]   two schedulers on two threads
//   p495_sched_timing_test async [graphs] [delay_us] mock async backend + CPU
//
// mix:   every logged line must report exactly `graphs / every` calls. Shared
//        state would mix the counters of the two threads.
// async: the mock backend returns before its work is done. The submit must show
//        up as compute_us near zero and the wait must show up as sync_us on the
//        next split, not as execution time of the mock.
//
// The checks are done on the logged lines by the caller.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

// ------------------------------------------------------------------ mock backend

struct mock_ctx {
    ggml_backend_t cpu      = nullptr;
    int            delay_us = 0;
    std::thread    worker;
};

static const char * mock_get_name(ggml_backend_t backend) {
    (void) backend;
    return "MOCK";
}

static void mock_free(ggml_backend_t backend) {
    mock_ctx * ctx = (mock_ctx *) backend->context;
    if (ctx->worker.joinable()) {
        ctx->worker.join();
    }
    delete ctx;
    delete backend;
}

static enum ggml_status mock_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    mock_ctx * ctx = (mock_ctx *) backend->context;
    if (ctx->worker.joinable()) {
        ctx->worker.join();
    }
    ctx->worker = std::thread([ctx, cgraph]() {
        std::this_thread::sleep_for(std::chrono::microseconds(ctx->delay_us));
        ggml_backend_graph_compute(ctx->cpu, cgraph);
    });
    // returns while the work is still running, like a device queue submit
    return GGML_STATUS_SUCCESS;
}

static void mock_synchronize(ggml_backend_t backend) {
    mock_ctx * ctx = (mock_ctx *) backend->context;
    if (ctx->worker.joinable()) {
        ctx->worker.join();
    }
}

static ggml_backend_t mock_init(ggml_backend_t cpu, int delay_us) {
    static int mock_guid;

    ggml_backend_t backend = new ggml_backend();
    memset(backend, 0, sizeof(*backend));

    mock_ctx * ctx = new mock_ctx();
    ctx->cpu       = cpu;
    ctx->delay_us  = delay_us;

    backend->guid    = (ggml_guid_t) &mock_guid;
    backend->device  = ggml_backend_get_device(cpu);
    backend->context = ctx;

    backend->iface.get_name     = mock_get_name;
    backend->iface.free         = mock_free;
    backend->iface.graph_compute = mock_graph_compute;
    backend->iface.synchronize  = mock_synchronize;

    return backend;
}

// ------------------------------------------------------------------ mix mode

struct worker {
    int n_graphs;
    int n_nodes;
    int rc = 400;
};

static void run_worker(worker * w) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        return;
    }
    ggml_backend_t             be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t bt = ggml_backend_dev_buffer_type(dev);

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 2 + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return;
    }

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    ggml_tensor * out = a;
    for (int i = 0; i < w->n_nodes; i++) {
        out = ggml_add(ctx, out, b);
        ggml_set_name(out, "out");
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt);
    if (buf == nullptr) {
        ggml_free(ctx);
        return;
    }
    {
        float data[1024];
        for (int i = 0; i < 1024; i++) {
            data[i] = 0.001f * (float) i;
        }
        ggml_backend_tensor_set(a, data, 0, sizeof(data));
        ggml_backend_tensor_set(b, data, 0, sizeof(data));
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(&be, &bt, 1, 4096, false, false);
    int rc = 0;
    for (int i = 0; i < w->n_graphs && rc == 0; i++) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            rc = 1;
            break;
        }
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            rc = 2;
            break;
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    w->rc = rc;
}

static int run_mix(int n_graphs) {
    worker w1 = { n_graphs, 1 };
    worker w2 = { n_graphs, 3 };

    std::thread t1(run_worker, &w1);
    std::thread t2(run_worker, &w2);
    t1.join();
    t2.join();

    printf("P495SCHED mix done rc %d %d\n", w1.rc, w2.rc);
    return (w1.rc == 0 && w2.rc == 0) ? 0 : 1;
}

// ------------------------------------------------------------------ async mode

static int run_async(int n_graphs, int delay_us) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        printf("P495SCHED async no cpu device\n");
        return 1;
    }

    ggml_backend_t cpu  = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t mock = mock_init(cpu, delay_us);

    ggml_backend_buffer_type_t bt = ggml_backend_dev_buffer_type(dev);

    ggml_backend_t             backends[2] = { mock, cpu };
    ggml_backend_buffer_type_t bufts[2]    = { bt, bt };

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 2 + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return 1;
    }

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    ggml_tensor * t0 = ggml_add(ctx, a, b);        // mock, async
    ggml_set_name(t0, "t0");
    ggml_tensor * t1 = ggml_add(ctx, t0, b);       // cpu, waits for the mock
    ggml_set_name(t1, "t1");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t1);

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 256, false, false);

    float data[1024];
    for (int i = 0; i < 1024; i++) {
        data[i] = 0.001f * (float) i;
    }

    int rc = 0;
    for (int i = 0; i < n_graphs && rc == 0; i++) {
        ggml_backend_sched_reset(sched);
        // reset clears the assignment, so it has to be set again for every graph
        ggml_backend_sched_set_tensor_backend(sched, t0, mock);
        ggml_backend_sched_set_tensor_backend(sched, t1, cpu);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            rc = 1;
            break;
        }
        ggml_backend_tensor_set(a, data, 0, sizeof(data));
        ggml_backend_tensor_set(b, data, 0, sizeof(data));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            rc = 2;
            break;
        }
    }

    float out[1024] = { 0 };
    double max_err = 0.0;
    if (rc == 0) {
        ggml_backend_tensor_get(t1, out, 0, sizeof(out));
        for (int i = 0; i < 1024; i++) {
            const double want = 3.0 * (double) data[i];
            const double err  = out[i] > want ? (double) out[i] - want : want - (double) out[i];
            if (err > max_err) {
                max_err = err;
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(mock);
    ggml_backend_free(cpu);

    printf("P495SCHED async done rc %d graphs %d delay_us %d max_err %.9f\n", rc, n_graphs, delay_us, max_err);
    return rc == 0 ? 0 : 1;
}

// ------------------------------------------------------------------ main

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "mix";

    if (strcmp(mode, "mix") == 0) {
        const int n_graphs = argc > 2 ? atoi(argv[2]) : 300;
        const int every    = argc > 3 ? atoi(argv[3]) : 10;
        printf("P495SCHED mix graphs %d every %d\n", n_graphs, every);
        return run_mix(n_graphs);
    }
    if (strcmp(mode, "async") == 0) {
        const int n_graphs = argc > 2 ? atoi(argv[2]) : 20;
        const int delay_us = argc > 3 ? atoi(argv[3]) : 20000;
        printf("P495SCHED async graphs %d delay_us %d\n", n_graphs, delay_us);
        return run_async(n_graphs, delay_us);
    }

    printf("P495SCHED unknown mode %s\n", mode);
    return 2;
}

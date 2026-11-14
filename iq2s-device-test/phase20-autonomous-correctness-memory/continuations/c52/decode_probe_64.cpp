#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Storage read counter for one interval. read_bytes counts block layer reads.
static unsigned long long proc_read_bytes() {
    std::ifstream input("/proc/self/io");
    std::string key;
    unsigned long long value = 0;
    while (input >> key >> value) {
        if (key == "read_bytes:") {
            return value;
        }
    }
    return 0;
}

static void snapshot(const char * output, const char * phase) {
    std::ofstream stream(std::string(output) + "." + phase + ".telemetry");
    stream << "MONOTONIC_BEGIN_MS " << now_ms() << '\n';
    for (const char * path : {"/proc/uptime", "/proc/meminfo", "/proc/vmstat", "/proc/self/status", "/proc/self/stat", "/proc/self/smaps_rollup"}) {
        std::ifstream input(path);
        stream << "FILE " << path << '\n' << input.rdbuf() << '\n';
    }
    stream << "MONOTONIC_END_MS " << now_ms() << '\n';
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 8) {
        std::fprintf(stderr, "usage: decode-probe MODEL OVERRIDE_FILE STEPS LOGITS_FILE [none|mmap] [HTP0:0|CPU] [GPU_LAYERS]\n");
        return 2;
    }
    const int steps = std::atoi(argv[3]);
    // Phase 52: was 32. n_ctx is 128 below and the prompt is a few tokens, so
    // 64 steps fit. Raising the limit does not touch the decode loop, which the
    // 32-step regression against the known hashes checks before 64 is used.
    if (steps < 1 || steps > 64) {
        return 2;
    }
    const std::string mode = argc >= 6 ? argv[5] : "none";
    if (mode != "none" && mode != "mmap") {
        return 2;
    }
    std::ifstream config(argv[2]);
    if (!config) {
        return 2;
    }
    // One "regex=CPU" or "regex=HTP" rule per line. Rules are applied in order
    // and the first match wins, so a following line can exempt a tensor from a
    // broader rule above it.
    std::vector<llama_model_tensor_buft_override> overrides;
    std::vector<std::string> patterns;
    std::vector<bool> targets_cpu;
    {
        std::string line;
        while (std::getline(config, line)) {
            while (!line.empty() && (line.back() == 13 || line.back() == ' ' || line.back() == '	')) {
                line.pop_back();
            }
            if (line.empty() || line[0] == '#') {
                continue;
            }
            bool cpu = true;
            if (line.size() >= 4 && line.compare(line.size() - 4, 4, "=CPU") == 0) {
                line.resize(line.size() - 4);
            } else if (line.size() >= 4 && line.compare(line.size() - 4, 4, "=HTP") == 0) {
                line.resize(line.size() - 4);
                cpu = false;
            } else {
                return 2;
            }
            patterns.push_back(line);
            targets_cpu.push_back(cpu);
        }
    }
    if (patterns.empty() || patterns.size() >= llama_max_tensor_buft_overrides()) {
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (std::getenv("C22_TRACE")) {
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level <= GGML_LOG_LEVEL_DEBUG) { std::fputs(text, stderr); }
        }, nullptr);
    }
    llama_backend_init();
    const std::string device_name = argc >= 7 ? argv[6] : "HTP0:0";
    if (device_name != "HTP0:0" && device_name != "CPU") {
        return 2;
    }
    auto device = ggml_backend_dev_by_name(device_name.c_str());
    if (!device) {
        return 3;
    }
    const bool cpu_only = device_name == "CPU";
    ggml_backend_dev_t devices[] = { cpu_only ? nullptr : device, nullptr };
    const ggml_backend_buffer_type_t htp_buft = cpu_only ? nullptr : ggml_backend_dev_buffer_type(device);
    for (size_t i = 0; i < patterns.size(); ++i) {
        overrides.push_back({ patterns[i].c_str(), targets_cpu[i] ? ggml_backend_cpu_buffer_type() : htp_buft });
    }
    overrides.push_back({ nullptr, nullptr });
    auto mp = llama_model_default_params();
    mp.devices = devices;
    const int gpu_layers = argc == 8 ? std::atoi(argv[7]) : -1;
    if (gpu_layers < -1 || gpu_layers > 65) {
        return 2;
    }
    mp.n_gpu_layers = cpu_only ? 0 : gpu_layers;
    mp.load_mode = mode == "mmap" ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    mp.tensor_buft_overrides = overrides.data();
    if (std::getenv("P19_NO_EXTRA_BUFTS")) {
        mp.use_extra_bufts = false;
    }
    snapshot(argv[4], "load-begin");
    const double load_begin = now_ms();
    std::printf("PHASE load_begin %.3f\n", load_begin);
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 4;
    }
    std::printf("PHASE load_end %.3f load_ms %.3f\n", now_ms(), now_ms() - load_begin);
    snapshot(argv[4], "load-end");
    for (const char * name : {"maps", "smaps"}) {
        std::ifstream input(std::string("/proc/self/") + name);
        std::ofstream snapshot(std::string(argv[4]) + "." + name);
        snapshot << input.rdbuf();
    }

    // Phase 27 handshake: the model is loaded and the HTP buffer is allocated.
    // Block here until the runner reports that its post-load warm-up finished,
    // so prefill cannot start first.
    if (const char * resume_path = std::getenv("P27_RESUME_FILE")) {
        std::printf("PHASE warm_wait_begin %.3f\n", now_ms());
        std::fflush(stdout);
        bool warmed = false;
        for (int i = 0; i < 900; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::ifstream f(resume_path);
            if (!f) {
                continue;
            }
            std::string line;
            std::getline(f, line);
            if (line.rfind("WARM_DONE", 0) == 0) { warmed = true; break; }
            if (line.rfind("WARM_FAIL", 0) == 0) { break; }
        }
        if (!warmed) {
            std::printf("PHASE warm_abort %.3f\n", now_ms());
            llama_model_free(model);
            return 6;
        }
        std::printf("PHASE warm_wait_end %.3f\n", now_ms());
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 32;
    cp.n_ubatch = 32;
    // Phase 44: thread count for the CPU side is now selectable at run time.
    int n_threads = 6;
    if (const char * s = std::getenv("P44_NTHREADS")) {
        const int v = std::atoi(s);
        if (v > 0 && v <= 16) {
            n_threads = v;
        }
    }
    std::printf("THREADS %d\n", n_threads);
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = !cpu_only;
    cp.op_offload = !cpu_only;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 5;
    }
    const auto vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> replay;
    if (const char * path = std::getenv("P19_REPLAY_TOKENS")) {
        std::ifstream input(path);
        int64_t token;
        while (input >> token) {
            if (token < 0 || token >= n_vocab) {
                return 2;
            }
            replay.push_back((llama_token) token);
        }
        if ((int) replay.size() != steps) {
            return 2;
        }
    }
    const std::string prompt = "The capital of Japan is";
    std::vector<llama_token> tokens(64);
    const int n_prompt = llama_tokenize(vocab, prompt.data(), (int) prompt.size(), tokens.data(), (int) tokens.size(), true, false);
    if (n_prompt <= 0 || n_prompt > (int) tokens.size()) {
        llama_free(ctx);
        llama_model_free(model);
        return 6;
    }
    tokens.resize(n_prompt);
    std::printf("PROMPT_IDS");
    for (auto token : tokens) {
        std::printf(" %d", token);
    }
    std::printf("\n");
    snapshot(argv[4], "prefill-begin");
    std::printf("PHASE prefill_begin %.3f\n", now_ms());
    const double prefill_begin = now_ms();
    int status = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prompt));
    std::printf("PHASE prefill_end %.3f prefill_ms %.3f status %d\n", now_ms(), now_ms() - prefill_begin, status);
    snapshot(argv[4], "prefill-end");

    // Second halt point: after prefill, before the first decode. The runner can
    // measure residency, restore only the pages the prefill evicted, and then
    // let decode start.
    if (const char * resume2 = std::getenv("P27_RESUME2_FILE")) {
        std::printf("PHASE pause2_begin %.3f\n", now_ms());
        std::fflush(stdout);
        bool ok2 = false;
        for (int i = 0; i < 600; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::ifstream f(resume2);
            if (!f) {
                continue;
            }
            std::string line;
            std::getline(f, line);
            if (line.rfind("RESUME2_OK", 0) == 0) { ok2 = true; break; }
            if (line.rfind("RESUME2_SKIP", 0) == 0) { break; }
        }
        std::printf("PHASE pause2_end %.3f resumed %d\n", now_ms(), ok2 ? 1 : 0);
        std::fflush(stdout);
    }
    FILE * output = status == 0 ? std::fopen(argv[4], "wb") : nullptr;
    if (!output) {
        llama_free(ctx);
        llama_model_free(model);
        return 7;
    }
    std::vector<double> decode_times;
    const double generation_begin = now_ms();
    for (int step = 0; step < steps; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits || !std::all_of(logits, logits + n_vocab, [](float x) { return std::isfinite(x); })) {
            status = 8;
            break;
        }
        if (std::fwrite(logits, sizeof(float), n_vocab, output) != (size_t) n_vocab) {
            status = 9;
            break;
        }
        llama_token token = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
        std::printf("TOKEN step %d id %d logit %.9g finite %d\n", step, token, logits[token], n_vocab);
        if (!replay.empty()) {
            token = replay[step];
            std::printf("REPLAY step %d input_id %d\n", step, token);
        }
        const double begin = now_ms();
        const unsigned long long io_begin = proc_read_bytes();
        status = llama_decode(ctx, llama_batch_get_one(&token, 1));
        const unsigned long long io_end = proc_read_bytes();
        const double elapsed = now_ms() - begin;
        std::printf("DECODE step %d ms %.6f status %d\n", step, elapsed, status);
        std::printf("IO step %d read_bytes %llu\n", step, io_end - io_begin);
        if (status != 0) {
            break;
        }
        decode_times.push_back(elapsed);
    }
    if (std::fclose(output) != 0) {
        status = 10;
    }
    const double generation_ms = now_ms() - generation_begin;
    snapshot(argv[4], "generation-end");
    if (status == 0) {
        const float * final_logits = llama_get_logits_ith(ctx, -1);
        if (!final_logits || !std::all_of(final_logits, final_logits + n_vocab, [](float x) { return std::isfinite(x); })) {
            status = 8;
        }
    }
    if (status == 0 && !decode_times.empty()) {
        double sum = 0;
        for (double elapsed : decode_times) {
            sum += elapsed;
        }
        std::printf("RESULT forwards %zu decode_ms %.6f tokens_s %.9f\n", decode_times.size(), sum / decode_times.size(), 1000 * decode_times.size() / sum);
        std::printf("GENERATION forwards %zu wall_ms %.6f tokens_s %.9f\n", decode_times.size(), generation_ms, 1000 * decode_times.size() / generation_ms);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return status == 0 ? 0 : 1;
}

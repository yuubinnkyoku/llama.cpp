// Phase 53 probe.
//
// Derived from the Phase 52 probe (continuations/c52/decode_probe_64.cpp), which
// itself is the Phase 20 c27 probe with only the step limit raised. The Phase 53
// additions are strictly opt-in through environment variables, so with none of
// them set the program behaves exactly like the Phase 52 probe and the known
// 32-step hashes still have to match before any new number is used.
//
//   P53_STEPS_MAX      step limit for the STEPS argument (default 64, max 4096)
//   P53_N_CTX          context size (default 128); auto-raised and reported if
//                      prompt + steps do not fit
//   P53_N_BATCH        logical batch size (default 32)
//   P53_N_UBATCH       micro batch size (default 32)
//   P53_PROMPT_FILE    file with the prompt text
//   P53_PROMPT_TOKENS  exact prompt length in tokens; the text is repeated
//                      until it is long enough, then truncated
//   P53_SEGMENTS       number of prompt+decode segments (default 1); from the
//                      second segment on the KV cache is cleared and the prompt
//                      is decoded again, so a long sustained run keeps a small
//                      context and does not need a large KV allocation
//   P53_TEMP           sampling temperature (default 0 = greedy argmax, the
//                      Phase 52 behaviour)
//   P53_TOP_K/P53_TOP_P  optional filters for the temperature path
//   P53_SEED           seed for the dist sampler (default 0)
//   P53_NO_LOGITS      do not write the per-step logits (needed for long runs:
//                      500 steps x 248320 float is about 0.5 GB)
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

static int env_int(const char * name, int fallback, int low, int high) {
    if (const char * s = std::getenv(name)) {
        const int v = std::atoi(s);
        if (v >= low && v <= high) {
            return v;
        }
    }
    return fallback;
}

// The default prompt text. It only has to be long enough that repeating it a
// few times covers P53_PROMPT_TOKENS; the exact content is not the subject of
// the measurement.
static const char * kDefaultPromptText =
    "The study of language models on mobile hardware raises a question that is "
    "easy to state and hard to answer. A phone has a fixed thermal budget, a "
    "battery that the user expects to last a day, and a memory hierarchy that "
    "was designed for applications that touch a few hundred megabytes at most. "
    "A twenty seven billion parameter network in a two bit format still needs "
    "close to nine gigabytes of weights, so every token that the device "
    "produces has to stream a large part of the model through the caches. The "
    "interesting part is not whether this can be done once. It is whether it "
    "can be done repeatedly, at a rate that a person would accept, without the "
    "device becoming uncomfortable to hold or the operating system deciding "
    "that the process should be reclaimed. ";

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 8) {
        std::fprintf(stderr, "usage: decode-probe MODEL OVERRIDE_FILE STEPS LOGITS_FILE [none|mmap] [HTP0:0|CPU] [GPU_LAYERS]\n");
        return 2;
    }
    const int steps_max = env_int("P53_STEPS_MAX", 64, 1, 4096);
    const int steps = std::atoi(argv[3]);
    // Phase 52 raised this from 32 to 64 with n_ctx 128 and a five token
    // prompt. Phase 53 makes the limit a variable so a sustained run can decode
    // far more than 64 tokens per segment, but the default stays 64 so the
    // Phase 52 regression is unchanged.
    if (steps < 1 || steps > steps_max) {
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
            while (!line.empty() && (line.back() == 13 || line.back() == ' ' || line.back() == '\t')) {
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
    int n_ctx = env_int("P53_N_CTX", 128, 64, 8192);
    int n_batch = env_int("P53_N_BATCH", 32, 1, 8192);
    const int n_ubatch = env_int("P53_N_UBATCH", 32, 1, 8192);
    // This llama.cpp version requires every llama_decode call to carry at most
    // n_batch tokens (llama-context.cpp GGML_ASSERT(n_tokens_all <= n_batch)); a
    // 128 token prompt with the default n_batch of 32 aborts. Both the context
    // and the batch are therefore derived from the requested prompt lengths.
    const int prompt_tokens_wanted = env_int("P53_PROMPT_TOKENS", 0, 0, 8192);
    int longest_decode = prompt_tokens_wanted > 0 ? prompt_tokens_wanted : 5;
    {
        int listed = 0;
        if (const char * s = std::getenv("P53_PROMPT_LIST")) {
            const char * p = s;
            while (*p) {
                while (*p == ',' || *p == ' ') { ++p; }
                if (*p < '0' || *p > '9') { break; }
                int v = 0;
                while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; }
                if (v > longest_decode && v <= 8192) {
                    longest_decode = v;
                }
                ++listed;
            }
        }
        if (longest_decode < 1) {
            longest_decode = 1;
        }
    }
    if (n_batch < longest_decode) {
        std::printf("NBATCH_AUTO from %d to %d for longest decode %d\n", n_batch, longest_decode, longest_decode);
        n_batch = longest_decode;
    }
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_ubatch;
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
    const int segments = env_int("P53_SEGMENTS", 1, 1, 256);
    const bool no_logits = std::getenv("P53_NO_LOGITS") != nullptr;
    // Auto-raise the context when the caller asked for a long prompt but left
    // the default context size. Reported, never silent.
    {
        const int need = longest_decode + steps + 16;
        if (need > n_ctx) {
            const int raised = need > 8192 ? 8192 : need;
            std::printf("NCTX_AUTO from %d to %d for prompt %d steps %d\n", n_ctx, raised, longest_decode, steps);
            n_ctx = raised;
            cp.n_ctx = n_ctx;
        }
    }
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 5;
    }
    const auto vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::printf("CTX n_ctx %d n_batch %d n_ubatch %d segments %d steps %d\n", n_ctx, n_batch, n_ubatch, segments, steps);
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
    // Prompt. With no Phase 53 variable set this is byte for byte the Phase 52
    // path, so the regression token ids are unchanged.
    std::vector<llama_token> tokens;
    std::vector<llama_token> token_pool;
    if (prompt_tokens_wanted > 0) {
        std::string text;
        if (const char * path = std::getenv("P53_PROMPT_FILE")) {
            std::ifstream input(path);
            if (!input) {
                llama_free(ctx);
                llama_model_free(model);
                return 2;
            }
            text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        } else {
            text = kDefaultPromptText;
        }
        if (text.empty()) {
            llama_free(ctx);
            llama_model_free(model);
            return 2;
        }
        const std::string unit = text;
        while ((int) text.size() < longest_decode * 8 && text.size() < 1024u * 1024u) {
            text += unit;
        }
        // One token consumes at least one byte, so text.size() + 16 is a safe
        // upper bound for the token count and the truncation below does the
        // rest. An exact-size buffer failed here on the first attempt: the
        // padded text tokenises to more tokens than prompt_tokens_wanted and
        // llama_tokenize then returns minus that count.
        std::vector<llama_token> buffer(text.size() + 16);
        const int n_prompt = llama_tokenize(vocab, text.data(), (int) text.size(), buffer.data(), (int) buffer.size(), true, false);
        if (n_prompt < longest_decode) {
            std::printf("PROMPT_SHORT got %d want %d\n", n_prompt, longest_decode);
            llama_free(ctx);
            llama_model_free(model);
            return 6;
        }
        buffer.resize(n_prompt);
        tokens.assign(buffer.begin(), buffer.begin() + prompt_tokens_wanted);
        // Kept for the prefill sweep below: the full tokenisation, truncated to
        // whatever length the sweep asks for.
        token_pool = buffer;
    } else {
        const std::string prompt = "The capital of Japan is";
        std::vector<llama_token> buffer(64);
        const int n_prompt = llama_tokenize(vocab, prompt.data(), (int) prompt.size(), buffer.data(), (int) buffer.size(), true, false);
        if (n_prompt <= 0 || n_prompt > (int) buffer.size()) {
            llama_free(ctx);
            llama_model_free(model);
            return 6;
        }
        buffer.resize(n_prompt);
        tokens = buffer;
    }
    const int n_prompt = (int) tokens.size();
    std::printf("PROMPT_IDS");
    for (auto token : tokens) {
        std::printf(" %d", token);
    }
    std::printf("\n");
    std::printf("PROMPT_LEN %d\n", n_prompt);

    // Prefill sweep. One model load can then measure prompt processing at
    // several lengths, which matters twice: a 27B load costs more than a minute
    // and churns 9 GB of page cache, and each extra load pushes the run into
    // the swap guard. Every length is measured from a cleared memory so a short
    // prompt cannot ride on a warm cache from the previous one.
    if (const char * list = std::getenv("P53_PROMPT_LIST")) {
        if (token_pool.empty()) {
            std::printf("SWEEP_SKIP empty token pool (P53_PROMPT_TOKENS must be > 0)\n");
        } else {
            const char * p = list;
            // One throwaway decode first: the very first llama_decode builds the
            // ggml graph and reserves the scheduler buffers, which costs seconds
            // and would otherwise be charged to the first sweep length and make
            // the slope meaningless.
            {
                llama_memory_clear(llama_get_memory(ctx), true);
                const double begin = now_ms();
                const int rc = llama_decode(ctx, llama_batch_get_one(token_pool.data(), 1));
                std::printf("SWEEP_WARM ms %.6f status %d\n", now_ms() - begin, rc);
                if (rc != 0) {
                    llama_free(ctx);
                    llama_model_free(model);
                    return 7;
                }
            }
            while (*p) {
                while (*p == ',' || *p == ' ') { ++p; }
                if (*p < '0' || *p > '9') { break; }
                int v = 0;
                while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; }
                if (v < 1 || v > (int) token_pool.size()) {
                    std::printf("SWEEP_SKIP len %d pool %zu\n", v, token_pool.size());
                    continue;
                }
                llama_memory_clear(llama_get_memory(ctx), true);
                const double begin = now_ms();
                const int rc = llama_decode(ctx, llama_batch_get_one(token_pool.data(), v));
                const double elapsed = now_ms() - begin;
                std::printf("PREFILL_SWEEP len %d ms %.6f tokens_s %.6f status %d\n", v, elapsed, 1000.0 * v / elapsed, rc);
                if (rc != 0) {
                    break;
                }
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            std::printf("SWEEP_DONE\n");
        }
    }

    // Sampling. Temperature 0 keeps the Phase 52 argmax selection exactly.
    float temperature = 0.0f;
    if (const char * s = std::getenv("P53_TEMP")) {
        temperature = (float) std::atof(s);
    }
    llama_sampler * smpl = nullptr;
    if (temperature > 0.0001f) {
        auto sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        if (const char * s = std::getenv("P53_TOP_K")) {
            const int k = std::atoi(s);
            if (k > 0) {
                llama_sampler_chain_add(smpl, llama_sampler_init_top_k(k));
            }
        }
        if (const char * s = std::getenv("P53_TOP_P")) {
            const float p = (float) std::atof(s);
            if (p > 0.0f && p < 1.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(p, 1));
            }
        }
        const uint32_t seed = (uint32_t) env_int("P53_SEED", 0, 0, 2147483647);
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }
    std::printf("SAMPLER temp %.3f kind %s\n", temperature, smpl ? "chain" : "argmax");

    snapshot(argv[4], "prefill-begin");
    std::printf("PHASE prefill_begin %.3f\n", now_ms());
    const double first_begin = now_ms();
    double prefill_begin = first_begin;
    int status = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prompt));
    std::printf("PHASE prefill_end %.3f prefill_ms %.3f status %d\n", now_ms(), now_ms() - prefill_begin, status);
    std::printf("PREFILL seg %d tokens %d ms %.6f tokens_s %.6f\n", 0, n_prompt, now_ms() - prefill_begin,
                1000.0 * n_prompt / (now_ms() - prefill_begin));
    std::printf("TTFT_MS %.6f\n", now_ms() - first_begin);
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
    FILE * output = nullptr;
    if (status == 0) {
        output = no_logits ? std::fopen("/dev/null", "wb") : std::fopen(argv[4], "wb");
    }
    if (!output) {
        llama_free(ctx);
        llama_model_free(model);
        return 7;
    }
    std::vector<double> decode_times;
    std::vector<llama_token> generated;
    const double generation_begin = now_ms();
    int segment_done = 0;
    for (int seg = 0; seg < segments && status == 0; ++seg) {
        if (seg > 0) {
            llama_memory_clear(llama_get_memory(ctx), true);
            prefill_begin = now_ms();
            status = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prompt));
            std::printf("PREFILL seg %d tokens %d ms %.6f tokens_s %.6f status %d\n", seg, n_prompt, now_ms() - prefill_begin,
                        1000.0 * n_prompt / (now_ms() - prefill_begin), status);
            if (status != 0) {
                break;
            }
        }
        std::vector<double> seg_times;
        for (int step = 0; step < steps; ++step) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits || !std::all_of(logits, logits + n_vocab, [](float x) { return std::isfinite(x); })) {
                status = 8;
                break;
            }
            if (!no_logits && std::fwrite(logits, sizeof(float), n_vocab, output) != (size_t) n_vocab) {
                status = 9;
                break;
            }
            llama_token token;
            if (smpl) {
                token = llama_sampler_sample(smpl, ctx, -1);
                llama_sampler_accept(smpl, token);
            } else {
                token = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
            }
            std::printf("TOKEN step %d id %d logit %.9g finite %d\n", step, token, logits[token], n_vocab);
            generated.push_back(token);
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
            std::printf("STEPT seg %d step %d t_ms %.3f\n", seg, step, now_ms());
            if (status != 0) {
                break;
            }
            decode_times.push_back(elapsed);
            seg_times.push_back(elapsed);
        }
        if (!seg_times.empty()) {
            double sum = 0;
            for (double elapsed : seg_times) {
                sum += elapsed;
            }
            std::printf("RESULT_SEG seg %d forwards %zu decode_ms %.6f tokens_s %.9f\n", seg, seg_times.size(), sum / seg_times.size(),
                        1000 * seg_times.size() / sum);
        }
        segment_done = seg + 1;
    }
    if (std::fclose(output) != 0) {
        status = 10;
    }
    const double generation_ms = now_ms() - generation_begin;
    snapshot(argv[4], "generation-end");
    // Detokenised output, so a free generation sample is kept as text as well as
    // ids. Small files, written next to the logits argument.
    {
        std::ofstream text_out(std::string(argv[4]) + ".gen.txt");
        std::string piece(256, '\0');
        for (auto token : generated) {
            const int n = llama_token_to_piece(vocab, token, piece.data(), (int) piece.size(), 0, false);
            if (n > 0) {
                text_out.write(piece.data(), n);
            }
        }
        text_out << '\n';
        std::ofstream ids_out(std::string(argv[4]) + ".gen.ids");
        for (auto token : generated) {
            ids_out << token << ' ';
        }
        ids_out << '\n';
    }
    std::printf("GENERATED n %zu\n", generated.size());
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
        std::printf("SEGMENTS done %d of %d\n", segment_done, segments);
    }
    if (smpl) {
        llama_sampler_free(smpl);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return status == 0 ? 0 : 1;
}

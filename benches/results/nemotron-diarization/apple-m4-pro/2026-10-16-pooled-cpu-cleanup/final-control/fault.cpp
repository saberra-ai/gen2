// Public C ABI failure-path control. No weights or frontend/inference execute.
#include <mach/mach.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "nemo_speech/asr.h"
#include "nemo_speech/diar.h"

static unsigned thread_count() {
    thread_act_array_t threads = nullptr;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS)
        throw std::runtime_error("cannot inspect own task threads");
    for (unsigned index = 0; index < count; ++index)
        mach_port_deallocate(mach_task_self(), threads[index]);
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), count * sizeof(thread_t));
    return count;
}

int main(int argc, char** argv) {
    std::vector<unsigned> immediate_counts, counts, waits;
    try {
        if (argc != 3 || std::filesystem::exists(argv[2]))
            throw std::runtime_error("expected exact runtime version and missing model path");
        const char* version = nemo_speech_asr_version();
        if (!version || std::strcmp(version, argv[1]))
            throw std::runtime_error("runtime version differs");
        nemo_speech_diar_model_config config{};
        config.size = sizeof(config);
        config.model_path = argv[2];
        config.gpu = -1;
        unsigned baseline = 0;
        for (unsigned index = 0; index < 101; ++index) {
            auto* model = reinterpret_cast<nemo_speech_diar_model*>(uintptr_t{1});
            const auto status = nemo_speech_diar_create(&config, &model);
            const char* error = nemo_speech_asr_last_error();
            if (status == NEMO_SPEECH_ASR_OK || model || !error ||
                std::string(error) != "Failed to load GGML file: " + std::string(argv[2]))
                throw std::runtime_error("failed model load did not clear handle and report file error");
            immediate_counts.push_back(thread_count());
            // pthread_join has returned, but task_threads can briefly include
            // retiring kernel threads. Preserve the immediate observation and
            // allow bounded settling before checking persistent worker growth.
            unsigned requested_wait = 0;
            unsigned observed = 0;
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                requested_wait += 10;
                observed = thread_count();
            } while (index && observed > baseline + 2 && requested_wait < 100);
            counts.push_back(observed);
            waits.push_back(requested_wait);
            if (index == 4) throw std::runtime_error("forced observation diagnostic");
            if (!index) baseline = counts.back();
            // Allow at most two incidental system workers, never a leaked
            // three-worker CPU pool across these100additional failures.
            if (counts.back() > baseline + 2) {
                throw std::runtime_error("failed model loads accumulate native threads after settling; baseline="
                    + std::to_string(baseline) + " observed=" + std::to_string(counts.back()));
            }
        }
        std::cout << "{\"failed_model_construction_attempts\":101,\"valid_models_loaded\":0,"
                     "\"frontend_or_neural_calls\":0,\"thread_growth_allowance\":2,"
                     "\"maximum_requested_settling_wait_milliseconds\":100,\"immediate_thread_counts\":[";
        for (unsigned index = 0; index < immediate_counts.size(); ++index)
            std::cout << (index ? "," : "") << immediate_counts[index];
        std::cout << "],\"requested_settling_wait_milliseconds\":[";
        for (unsigned index = 0; index < waits.size(); ++index)
            std::cout << (index ? "," : "") << waits[index];
        std::cout << "],\"threads_after_each_failure\":[";
        for (unsigned index = 0; index < counts.size(); ++index)
            std::cout << (index ? "," : "") << counts[index];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "failed_load_thread_observations: immediate=";
        for (auto value : immediate_counts) std::cerr << value << ',';
        std::cerr << " settled=";
        for (auto value : counts) std::cerr << value << ',';
        std::cerr << " requested_wait_ms=";
        for (auto value : waits) std::cerr << value << ',';
        std::cerr << '\n';
        std::cerr << error.what() << '\n';
        return 1;
    }
}

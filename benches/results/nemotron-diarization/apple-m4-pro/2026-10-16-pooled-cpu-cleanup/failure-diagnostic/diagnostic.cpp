// Public C ABI failure-path control. No weights or frontend/inference execute.
#include <mach/mach.h>
#include <cstdint>
#include <thread>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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
        std::vector<unsigned> counts;
        unsigned baseline = 0;
        for (unsigned index = 0; index < 101; ++index) {
            auto* model = reinterpret_cast<nemo_speech_diar_model*>(uintptr_t{1});
            const auto status = nemo_speech_diar_create(&config, &model);
            const char* error = nemo_speech_asr_last_error();
            if (status == NEMO_SPEECH_ASR_OK || model || !error ||
                std::string(error) != "Failed to load GGML file: " + std::string(argv[2]))
                throw std::runtime_error("failed model load did not clear handle and report file error");
            counts.push_back(thread_count());
            if (!index) baseline = counts.back();
            // Allow at most two incidental system workers, never a leaked
            // three-worker CPU pool across these100additional failures.
            if (counts.back() > baseline + 2) {
                std::cerr << "[DEBUG-pool-cleanup] failure_index=" << index << " baseline=" << baseline << " count=" << counts.back() << " history=";
                for (auto value : counts) std::cerr << value << ",";
                std::cerr << "\n";
                for (int tick = 0; tick < 10; ++tick) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    std::cerr << "[DEBUG-pool-cleanup] quiet_millis=" << (tick + 1) * 10 << " count=" << thread_count() << "\n";
                }
                thread_act_array_t threads = nullptr; mach_msg_type_number_t total = 0;
                if (task_threads(mach_task_self(), &threads, &total) == KERN_SUCCESS) {
                    for (unsigned k = 0; k < total; ++k) {
                        thread_extended_info_data_t info{}; mach_msg_type_number_t size = THREAD_EXTENDED_INFO_COUNT;
                        auto status = thread_info(threads[k], THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&info), &size);
                        std::cerr << "[DEBUG-pool-cleanup] thread=" << k << " status=" << status << " name=" << info.pth_name << " run_state=" << info.pth_run_state << "\n";
                        mach_port_deallocate(mach_task_self(), threads[k]);
                    }
                    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), total * sizeof(thread_t));
                }
                throw std::runtime_error("failed model loads accumulate native threads");
            }
        }
        std::cout << "{\"failed_model_construction_attempts\":101,\"valid_models_loaded\":0,"
                     "\"frontend_or_neural_calls\":0,\"thread_growth_allowance\":2,\"threads_after_each_failure\":[";
        for (unsigned index = 0; index < counts.size(); ++index)
            std::cout << (index ? "," : "") << counts[index];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

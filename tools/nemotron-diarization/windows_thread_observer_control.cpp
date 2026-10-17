// Actual Windows observer control; no NeMo imports, model or inference calls.
#include "windows_own_threads.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

class KnownWorker {
public:
    std::atomic<DWORD> id{0};
private:
    std::atomic<bool> stop_{false};
    std::thread worker_;
public:
    KnownWorker() : worker_([this] {
        id.store(GetCurrentThreadId());
        while (!stop_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }) {}
    void join() { stop_.store(true); if (worker_.joinable()) worker_.join(); }
    ~KnownWorker() { join(); }
};

int main() {
    try {
        const auto baseline = gen2_windows::own_thread_ids();
        const auto main_id = GetCurrentThreadId();
        KnownWorker worker;
        unsigned ready_wait = 0;
        while (!worker.id.load() && ready_wait < 1000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ready_wait += 10;
        }
        const auto worker_id = worker.id.load();
        const auto active = gen2_windows::own_thread_ids();
        if (!worker_id || worker_id == main_id
                || !std::binary_search(active.begin(), active.end(), worker_id)
                || !std::binary_search(active.begin(), active.end(), main_id))
            throw std::runtime_error("known worker/main thread not observed");
        worker.join();
        auto settled = gen2_windows::own_thread_ids();
        unsigned wait = 0;
        while (std::binary_search(settled.begin(), settled.end(), worker_id) && wait < 100) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait += 10;
            settled = gen2_windows::own_thread_ids();
        }
        if (std::binary_search(settled.begin(), settled.end(), worker_id)
                || !std::binary_search(settled.begin(), settled.end(), main_id))
            throw std::runtime_error("joined worker did not retire or main thread missing");
        std::cout << "{\"known_worker_observed\":true,\"known_worker_retired\":true,"
                     "\"baseline_threads\":" << baseline.size() << ",\"active_threads\":" << active.size()
                  << ",\"settled_threads\":" << settled.size() << ",\"requested_retirement_wait_ms\":" << wait
                  << ",\"valid_models_loaded\":0,\"frontend_or_neural_calls\":0}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

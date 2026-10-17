// Only own-process IDs leave the snapshot. Query/close failures are not counts.
#pragma once
#ifndef GEN2_THREAD_SNAPSHOT_FAKE_API
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#endif
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gen2_windows {
class ThreadSnapshot {
    HANDLE handle_;
public:
    ThreadSnapshot() : handle_(CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)) {
        if (handle_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("thread snapshot failed: " + std::to_string(GetLastError()));
    }
    ThreadSnapshot(const ThreadSnapshot&) = delete;
    ThreadSnapshot& operator=(const ThreadSnapshot&) = delete;
    ~ThreadSnapshot() { if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_); }
    HANDLE get() const { return handle_; }
    void close() {
        if (!CloseHandle(handle_))
            throw std::runtime_error("thread snapshot close failed: " + std::to_string(GetLastError()));
        handle_ = INVALID_HANDLE_VALUE;
    }
};

inline std::vector<std::uint32_t> own_thread_ids() {
    const DWORD process = GetCurrentProcessId();
    if (!process) throw std::runtime_error("invalid own process ID");
    ThreadSnapshot snapshot;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    if (!Thread32First(snapshot.get(), &entry))
        throw std::runtime_error("first thread query failed: " + std::to_string(GetLastError()));
    std::vector<std::uint32_t> ids;
    std::size_t entries = 0;
    while (true) {
        if (++entries > 1000000 || entry.dwSize > sizeof(entry)
                || entry.dwSize < offsetof(THREADENTRY32, th32OwnerProcessID) + sizeof(entry.th32OwnerProcessID))
            throw std::runtime_error("incomplete or excessive thread snapshot");
        if (entry.th32OwnerProcessID == process) {
            if (!entry.th32ThreadID) throw std::runtime_error("invalid own thread ID");
            ids.push_back(entry.th32ThreadID);
        }
        entry.dwSize = sizeof(entry);
        if (!Thread32Next(snapshot.get(), &entry)) {
            const DWORD error = GetLastError();
            if (error != ERROR_NO_MORE_FILES)
                throw std::runtime_error("next thread query failed: " + std::to_string(error));
            break;
        }
    }
    snapshot.close();
    std::sort(ids.begin(), ids.end());
    if (ids.empty() || std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::runtime_error("empty or duplicate own thread snapshot");
    return ids;
}
} // namespace gen2_windows

// Fake API regression only. This cannot qualify the actual Windows adapter.
#define GEN2_THREAD_SNAPSHOT_FAKE_API
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using DWORD = std::uint32_t;
using HANDLE = std::intptr_t;
constexpr HANDLE INVALID_HANDLE_VALUE = -1;
constexpr DWORD TH32CS_SNAPTHREAD = 4, ERROR_NO_MORE_FILES = 18;
struct THREADENTRY32 {
    DWORD dwSize, cntUsage, th32ThreadID, th32OwnerProcessID;
    long tpBasePri, tpDeltaPri;
    DWORD dwFlags;
};
struct State {
    std::vector<THREADENTRY32> rows;
    std::size_t position = 0;
    DWORD last_error = 0;
    bool open_error = false, first_error = false, next_error = false, short_row = false, close_error = false;
    unsigned opens = 0, closes = 0, close_attempts = 0;
} state;
DWORD GetCurrentProcessId() { return 7; }
DWORD GetLastError() { return state.last_error; }
HANDLE CreateToolhelp32Snapshot(DWORD flags, DWORD pid) {
    if (flags != TH32CS_SNAPTHREAD || pid != 0) throw std::runtime_error("wrong snapshot scope");
    if (state.open_error) { state.last_error = 5; return INVALID_HANDLE_VALUE; }
    ++state.opens;
    return 1;
}
bool CloseHandle(HANDLE handle) {
    if (handle != 1) throw std::runtime_error("wrong handle closed");
    ++state.close_attempts;
    if (state.close_error) { state.close_error = false; state.last_error = 5; return false; }
    ++state.closes;
    return true;
}
bool copy_row(THREADENTRY32* entry) {
    if (entry->dwSize != sizeof(THREADENTRY32)) throw std::runtime_error("dwSize not initialized/reset");
    *entry = state.rows.at(state.position);
    entry->dwSize = state.short_row ? 4 : sizeof(THREADENTRY32);
    return true;
}
bool Thread32First(HANDLE, THREADENTRY32* entry) {
    if (state.first_error || state.rows.empty()) { state.last_error = ERROR_NO_MORE_FILES; return false; }
    return copy_row(entry);
}
bool Thread32Next(HANDLE, THREADENTRY32* entry) {
    if (state.next_error) { state.last_error = 5; return false; }
    if (++state.position == state.rows.size()) { state.last_error = ERROR_NO_MORE_FILES; return false; }
    return copy_row(entry);
}
#include "windows_own_threads.h"

THREADENTRY32 row(DWORD process, DWORD thread) { return {sizeof(THREADENTRY32), 0, thread, process, 0, 0, 0}; }
void check(bool value) { if (!value) throw std::runtime_error("fake snapshot regression failed"); }
void reset() { state = State{}; state.rows = {row(9, 100), row(7, 3), row(8, 4), row(7, 2)}; }
void rejected() {
    bool failed = false;
    try { gen2_windows::own_thread_ids(); } catch (const std::runtime_error&) { failed = true; }
    check(failed && state.closes == state.opens);
}
int main() {
    try {
        reset(); check(gen2_windows::own_thread_ids() == std::vector<std::uint32_t>({2, 3}));
        check(state.opens == 1 && state.closes == 1 && state.close_attempts == 1);
        reset(); state.open_error = true; rejected();
        reset(); state.first_error = true; rejected();
        reset(); state.next_error = true; rejected();
        reset(); state.short_row = true; rejected();
        reset(); state.close_error = true; rejected(); check(state.close_attempts == 2);
        reset(); state.rows = {row(9, 100)}; rejected();
        reset(); state.rows = {row(7, 0)}; rejected();
        reset(); state.rows = {row(7, 2), row(7, 2)}; rejected();
        std::cout << "9 fake snapshot cases passed; no Windows API or model calls\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

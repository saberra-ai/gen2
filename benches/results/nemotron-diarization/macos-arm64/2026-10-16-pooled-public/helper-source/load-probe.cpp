// Qualification only: resolve the C ABI version without creating a model.
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::cerr << dlerror() << '\n';
        return 1;
    }
    auto version = reinterpret_cast<const char* (*)()>(dlsym(library, "nemo_speech_asr_version"));
    const char* actual = version ? version() : nullptr;
    const bool matches = actual && std::strcmp(actual, argv[2]) == 0;
    if (actual) std::cout << actual << '\n';
    dlclose(library);
    return matches ? 0 : 1;
}

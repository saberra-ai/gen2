#pragma once
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
#define G2W_API __declspec(dllexport)
#else
#define G2W_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Stable host shim, not the layout of whisper_full_params. One owner/thread per
// context. No callbacks may unwind and every pointer is borrowed for this call.
typedef int (*g2w_cancel)(void *);
G2W_API const char * gen2_whisper_abi(void);
G2W_API void * gen2_whisper_open(const void * model,size_t bytes);
// Additive profile extension: exact multilingual tiny-q5_1, host verifies SHA256.
// The legacy constructor remains base-only. Same inference strategy/limits.
G2W_API void * gen2_whisper_open_tiny_v1(const void * model,size_t bytes);
G2W_API void gen2_whisper_close(void * context);
// Input1800–480000 finite mono16kHz frames. 0=complete,1=failed,2=cancelled. Language0=auto,1=en,2=es; never translate.
G2W_API int gen2_whisper_run(void *,const float *,size_t,int,int,g2w_cancel,void *);
G2W_API int gen2_whisper_segments(void *);
G2W_API int gen2_whisper_language(void *);
// Copies UTF-8 without a terminator; writes byte count, start/end in10ms units.
G2W_API int gen2_whisper_segment(void *,int,int64_t *,int64_t *,char *,size_t,size_t *);
#ifdef __cplusplus
}
#endif

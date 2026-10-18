"""Windows-only thread CPU counters around the admitted batched wall observer."""
from ggml_matrix_batch_profile_source import transform as batch_transform
from ggml_cpu_profile_source import PATH,ORIGINAL_SHA256,replace_once

ENVIRONMENT='GEN2_DIAR_THREAD_CPU_PROFILE'
PREFIX='[DEBUG-thread-cpu-v1]'
MATRIX_PREFIX='[DEBUG-matrix-cpu-v1]'

HELPERS=br'''
#if !defined(_WIN32) && !defined(GEN2_THREAD_CPU_CLOCK_TEST)
#error Thread CPU profiling requires actual Windows APIs
#endif
static _Thread_local uint64_t gen2_clock_errors;
static _Thread_local unsigned gen2_clock_error_bits, gen2_clock_first_error;
static void gen2_clock_error(unsigned bits, unsigned code) {
    if (!gen2_clock_errors) gen2_clock_first_error = code;
    gen2_clock_errors++;
    gen2_clock_error_bits |= bits;
}
static uint64_t gen2_clock_cycles(void) {
    ULONG64 value = 0;
    if (!QueryThreadCycleTime(GetCurrentThread(), &value)) {
        gen2_clock_error(1, (unsigned) GetLastError());
        return 0;
    }
    return (uint64_t) value;
}
static uint64_t gen2_clock_delta(uint64_t before, uint64_t after) {
    if (after < before) {
        gen2_clock_error(4, 0);
        return 0;
    }
    return after - before;
}
struct gen2_thread_clock { uint64_t cycles, kernel, user; };
static struct gen2_thread_clock gen2_clock_thread(void) {
    FILETIME created = {0}, exited = {0}, kernel = {0}, user = {0};
    struct gen2_thread_clock value = {0};
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) {
        gen2_clock_error(2, (unsigned) GetLastError());
    } else {
        value.kernel = ((uint64_t) kernel.dwHighDateTime << 32) | kernel.dwLowDateTime;
        value.user = ((uint64_t) user.dwHighDateTime << 32) | user.dwLowDateTime;
    }
    value.cycles = gen2_clock_cycles();
    return value;
}
'''


def transform(raw):
    value=batch_transform(raw)
    def change(old,new):
        nonlocal value
        value=replace_once(value,old,new)
    change(b'struct gen2_matrix_stage { int calls, ok, failed; int64_t us; };',
        HELPERS+b'\nstruct gen2_matrix_stage { int calls, ok, failed; int64_t us; uint64_t cycles; };')
    change(b'    uint64_t chunks;\n};',b'    uint64_t chunks;\n    uint64_t conversion_cycles, inner_wait_cycles, chunk_cycles, kernel_cycles;\n};')
    change(b'    const int64_t start = record ? ggml_time_us() : 0;',
        b'    const uint64_t start_cycles = record ? gen2_clock_cycles() : 0;\n    const int64_t start = record ? ggml_time_us() : 0;')
    change(b'        value->us += ggml_time_us() - start;',
        b'        value->us += ggml_time_us() - start;\n        value->cycles += gen2_clock_delta(start_cycles, gen2_clock_cycles());')
    change(b'    int64_t gen2_conversion_start = 0;',
        b'    int64_t gen2_conversion_start = 0;\n    uint64_t gen2_conversion_cycles = 0;')
    change(b'gen2_conversion_start = ggml_time_us(); }',
        b'gen2_conversion_start = ggml_time_us(); gen2_conversion_cycles = gen2_clock_cycles(); }')
    change(b'    if (gen2_matrix_current && gen2_matrix_current->converted) gen2_matrix_current->conversion_us = ggml_time_us() - gen2_conversion_start;',
        b'    if (gen2_matrix_current && gen2_matrix_current->converted) {\n        gen2_matrix_current->conversion_us = ggml_time_us() - gen2_conversion_start;\n        gen2_matrix_current->conversion_cycles = gen2_clock_delta(gen2_conversion_cycles, gen2_clock_cycles());\n    }')
    change(b'    const int64_t gen2_wait_start = gen2_matrix_current ? ggml_time_us() : 0;',
        b'    const uint64_t gen2_wait_cycles = gen2_matrix_current ? gen2_clock_cycles() : 0;\n    const int64_t gen2_wait_start = gen2_matrix_current ? ggml_time_us() : 0;')
    change(b'    if (gen2_matrix_current) gen2_matrix_current->inner_wait_us = ggml_time_us() - gen2_wait_start;',
        b'    if (gen2_matrix_current) {\n        gen2_matrix_current->inner_wait_us = ggml_time_us() - gen2_wait_start;\n        gen2_matrix_current->inner_wait_cycles = gen2_clock_delta(gen2_wait_cycles, gen2_clock_cycles());\n    }')
    change(b'        const int64_t gen2_chunk_start = gen2_matrix_current ? ggml_time_us() : 0;',
        b'        const uint64_t gen2_chunk_cycles = gen2_matrix_current ? gen2_clock_cycles() : 0;\n        const int64_t gen2_chunk_start = gen2_matrix_current ? ggml_time_us() : 0;')
    change(b'        if (gen2_matrix_current) { gen2_matrix_current->chunks++; gen2_matrix_current->chunk_us += ggml_time_us() - gen2_chunk_start; }',
        b'        if (gen2_matrix_current) {\n            gen2_matrix_current->chunks++; gen2_matrix_current->chunk_us += ggml_time_us() - gen2_chunk_start;\n            gen2_matrix_current->chunk_cycles += gen2_clock_delta(gen2_chunk_cycles, gen2_clock_cycles());\n        }')
    change(b'    int64_t barrier_us;\n};',b'    int64_t barrier_us;\n    uint64_t kernel_cycles, barrier_cycles;\n};')
    change(b'getenv("GEN2_DIAR_MATRIX_PROFILE")',b'getenv("GEN2_DIAR_THREAD_CPU_PROFILE")')
    change(b'    int64_t gen2_started = 0;',b'    int64_t gen2_started = 0;\n    struct gen2_thread_clock gen2_cpu_begin = {0};\n    unsigned long gen2_thread_id = 0;')
    change(b'        gen2_started = ggml_time_us();',
        b'        gen2_clock_errors = 0; gen2_clock_error_bits = 0; gen2_clock_first_error = 0;\n        gen2_thread_id = (unsigned long) GetCurrentThreadId();\n        gen2_cpu_begin = gen2_clock_thread();\n        gen2_started = ggml_time_us();')
    change(b'        const int64_t gen2_before_kernel = gen2_profile ? ggml_time_us() : 0;',
        b'        const uint64_t gen2_before_kernel_cycles = gen2_profile ? gen2_clock_cycles() : 0;\n        const int64_t gen2_before_kernel = gen2_profile ? ggml_time_us() : 0;')
    change(b'        const int64_t gen2_after_kernel = gen2_profile ? ggml_time_us() : 0;',
        b'        const int64_t gen2_after_kernel = gen2_profile ? ggml_time_us() : 0;\n        const uint64_t gen2_kernel_cycles = gen2_profile ? gen2_clock_delta(gen2_before_kernel_cycles, gen2_clock_cycles()) : 0;')
    change(b'            gen2_matrix_current->kernel_us = gen2_after_kernel - gen2_before_kernel;',
        b'            gen2_matrix_current->kernel_us = gen2_after_kernel - gen2_before_kernel;\n            gen2_matrix_current->kernel_cycles = gen2_kernel_cycles;')
    change(b'            gen2_ops[node->op].kernel_us += gen2_after_kernel - gen2_before_kernel;',
        b'            gen2_ops[node->op].kernel_us += gen2_after_kernel - gen2_before_kernel;\n            gen2_ops[node->op].kernel_cycles += gen2_kernel_cycles;')
    change(b'        const int64_t gen2_before_outer = gen2_profile ? ggml_time_us() : 0;',
        b'        const uint64_t gen2_before_outer_cycles = gen2_profile ? gen2_clock_cycles() : 0;\n        const int64_t gen2_before_outer = gen2_profile ? ggml_time_us() : 0;')
    change(b'            gen2_ops[node->op].barrier_us += ggml_time_us() - gen2_before_outer;',
        b'            gen2_ops[node->op].barrier_us += ggml_time_us() - gen2_before_outer;\n            gen2_ops[node->op].barrier_cycles += gen2_clock_delta(gen2_before_outer_cycles, gen2_clock_cycles());')
    change(b'    const int64_t gen2_before_final = gen2_profile ? ggml_time_us() : 0;',
        b'    const uint64_t gen2_before_final_cycles = gen2_profile ? gen2_clock_cycles() : 0;\n    const int64_t gen2_before_final = gen2_profile ? ggml_time_us() : 0;')
    change(b'        const int64_t gen2_finished = ggml_time_us();',br'''        const int64_t gen2_finished = ggml_time_us();
        const uint64_t gen2_final_cycles = gen2_clock_delta(gen2_before_final_cycles, gen2_clock_cycles());
        const struct gen2_thread_clock gen2_cpu_end = gen2_clock_thread();
        const uint64_t gen2_graph_cycles = gen2_clock_delta(gen2_cpu_begin.cycles, gen2_cpu_end.cycles);
        const uint64_t gen2_kernel_100ns = gen2_clock_delta(gen2_cpu_begin.kernel, gen2_cpu_end.kernel);
        const uint64_t gen2_user_100ns = gen2_clock_delta(gen2_cpu_begin.user, gen2_cpu_end.user);
        fprintf(stderr, "[DEBUG-thread-cpu-v1] graph pool=%p generation=%d thread=%d tid=%lu cycles=%" PRIu64 " kernel_100ns=%" PRIu64 " user_100ns=%" PRIu64 " final_wait_cycles=%" PRIu64 " errors=%" PRIu64 " error_bits=%u first_error=%u\n",
                (void *) tp, gen2_generation, state->ith, gen2_thread_id, gen2_graph_cycles,
                gen2_kernel_100ns, gen2_user_100ns, gen2_final_cycles, gen2_clock_errors, gen2_clock_error_bits, gen2_clock_first_error);''')
    change(b'                        value->kernel_us, value->barrier_us);',br'''                        value->kernel_us, value->barrier_us);
                fprintf(stderr, "[DEBUG-thread-cpu-v1] op pool=%p generation=%d thread=%d id=%d kernel_cycles=%" PRIu64 " barrier_cycles=%" PRIu64 "\n",
                        (void *) tp, gen2_generation, state->ith, op, value->kernel_cycles, value->barrier_cycles);''')
    change(b'(size_t) gen2_matrix_count * 2048',b'(size_t) gen2_matrix_count * 4096')
    change(b'                gen2_used += (size_t) gen2_written;\n            }\n        }',br'''                gen2_used += (size_t) gen2_written;
            }
            if (!gen2_emission_error) {
                const int gen2_cpu_written = snprintf(gen2_buffer + gen2_used, gen2_capacity - gen2_used,
                        "[DEBUG-matrix-cpu-v1] matrix pool=%p generation=%d thread=%d node=%d first_cycles=%" PRIu64 " second_cycles=%" PRIu64 " conversion_cycles=%" PRIu64 " inner_wait_cycles=%" PRIu64 " chunk_cycles=%" PRIu64 " kernel_cycles=%" PRIu64 "\n",
                        (void *) tp, gen2_generation, state->ith, v->node, v->stage[0].cycles, v->stage[1].cycles,
                        v->conversion_cycles, v->inner_wait_cycles, v->chunk_cycles, v->kernel_cycles);
                if (gen2_cpu_written < 0 || (size_t) gen2_cpu_written >= gen2_capacity - gen2_used) gen2_emission_error = true;
                else gen2_used += (size_t) gen2_cpu_written;
            }
        }''')
    return value

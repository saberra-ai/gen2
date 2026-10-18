"""Diagnostic graph-only Windows CPU reads around the unchanged batched observer."""
from ggml_cpu_profile_source import ORIGINAL_SHA256, PATH, replace_once
from ggml_matrix_batch_profile_source import transform as batch_transform

ENVIRONMENT = 'GEN2_DIAR_GRAPH_CPU_PROFILE'
PREFIX = '[DEBUG-graph-cpu-v1]'

HELPERS = br'''
#if !defined(_WIN32) && !defined(GEN2_GRAPH_CPU_CLOCK_TEST)
#error Graph CPU profiling requires actual Windows APIs
#endif
static _Thread_local uint64_t gen2_graph_clock_errors;
static _Thread_local unsigned gen2_graph_clock_error_bits, gen2_graph_clock_first_error;
static void gen2_graph_clock_error(unsigned bits, unsigned code) {
    if (!gen2_graph_clock_errors) gen2_graph_clock_first_error = code;
    gen2_graph_clock_errors++;
    gen2_graph_clock_error_bits |= bits;
}
struct gen2_graph_clock { uint64_t kernel, user; };
static struct gen2_graph_clock gen2_graph_clock_read(void) {
    FILETIME created = {0}, exited = {0}, kernel = {0}, user = {0};
    struct gen2_graph_clock value = {0};
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) {
        gen2_graph_clock_error(1, (unsigned) GetLastError());
    } else {
        value.kernel = ((uint64_t) kernel.dwHighDateTime << 32) | kernel.dwLowDateTime;
        value.user = ((uint64_t) user.dwHighDateTime << 32) | user.dwLowDateTime;
    }
    return value;
}
static uint64_t gen2_graph_clock_delta(uint64_t before, uint64_t after) {
    if (after < before) {
        gen2_graph_clock_error(2, 0);
        return 0;
    }
    return after - before;
}
'''

DECLARATIONS = br'''
    struct gen2_graph_clock gen2_graph_begin = {0};
    unsigned long gen2_graph_tid = 0;
'''

BEGIN = br'''        gen2_graph_clock_errors = 0;
        gen2_graph_clock_error_bits = 0;
        gen2_graph_clock_first_error = 0;
        gen2_graph_tid = (unsigned long) GetCurrentThreadId();
        gen2_graph_begin = gen2_graph_clock_read();
'''

END = br'''
        const struct gen2_graph_clock gen2_graph_end = gen2_graph_clock_read();
        const uint64_t gen2_graph_kernel_100ns = gen2_graph_clock_delta(gen2_graph_begin.kernel, gen2_graph_end.kernel);
        const uint64_t gen2_graph_user_100ns = gen2_graph_clock_delta(gen2_graph_begin.user, gen2_graph_end.user);
        fprintf(stderr, "[DEBUG-graph-cpu-v1] graph pool=%p generation=%d thread=%d tid=%lu kernel_100ns=%" PRIu64 " user_100ns=%" PRIu64 " errors=%" PRIu64 " error_bits=%u first_error=%u\n",
                (void *) tp, gen2_generation, state->ith, gen2_graph_tid,
                gen2_graph_kernel_100ns, gen2_graph_user_100ns,
                gen2_graph_clock_errors, gen2_graph_clock_error_bits, gen2_graph_clock_first_error);
'''


def transform(raw):
    """Keep every wall region and inference call; add two CPU reads per graph."""
    value = batch_transform(raw)
    value = replace_once(value, b'struct gen2_matrix_stage {', HELPERS + b'\nstruct gen2_matrix_stage {')
    value = replace_once(value, b'getenv("GEN2_DIAR_MATRIX_PROFILE")',
                         b'getenv("GEN2_DIAR_GRAPH_CPU_PROFILE")')
    value = replace_once(value, b'    int64_t gen2_started = 0;',
                         b'    int64_t gen2_started = 0;' + DECLARATIONS)
    value = replace_once(value, b'        gen2_started = ggml_time_us();',
                         BEGIN + b'        gen2_started = ggml_time_us();')
    return replace_once(value, b'        const int64_t gen2_finished = ggml_time_us();',
                        b'        const int64_t gen2_finished = ggml_time_us();' + END)

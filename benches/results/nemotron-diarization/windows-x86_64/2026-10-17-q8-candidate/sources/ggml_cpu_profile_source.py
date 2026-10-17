"""Diagnostic-only timing around the pinned CPU loop's existing boundaries."""
import hashlib


PATH = 'ggml/src/ggml-cpu/ggml-cpu.c'
ORIGINAL_SHA256 = 'd4dd2c161a3f5c0ed45216683de99dd9b13b93330d81c992f4c4163fea904a04'
PREFIX = '[DEBUG-cpu-ops-v1]'
ENVIRONMENT = 'GEN2_DIAR_CPU_PROFILE'

HELPERS = br'''// Diagnostic-only: do not change graph planning, fusion or barriers.
struct gen2_cpu_op_timing {
    int roots;
    int fused;
    int64_t kernel_us;
    int64_t barrier_us;
};

static uint64_t gen2_cpu_profile_mix(uint64_t h, uint64_t value) {
    return (h ^ value) * UINT64_C(1099511628211);
}

static uint64_t gen2_cpu_profile_topology(const struct ggml_cgraph * graph) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (int i = 0; i < graph->n_nodes; ++i) {
        const struct ggml_tensor * node = graph->nodes[i];
        h = gen2_cpu_profile_mix(h, (uint64_t) node->op);
        h = gen2_cpu_profile_mix(h, (uint64_t) node->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h = gen2_cpu_profile_mix(h, (uint64_t) node->ne[d]);
        }
    }
    return h;
}

'''

INITIALIZE = br'''
    const char * gen2_profile_env = getenv("GEN2_DIAR_CPU_PROFILE");
    const bool gen2_profile = gen2_profile_env && strcmp(gen2_profile_env, "1") == 0;
    struct gen2_cpu_op_timing gen2_ops[GGML_OP_COUNT];
    int gen2_roots = 0;
    int gen2_fused = 0;
    int gen2_skipped = 0;
    int gen2_generation = 0;
    const int gen2_nodes = cgraph->n_nodes;
    int64_t gen2_started = 0;
    uint64_t gen2_topology = 0;
    uint64_t gen2_execution = UINT64_C(14695981039346656037);
    if (gen2_profile) {
        memset(gen2_ops, 0, sizeof(gen2_ops));
        gen2_generation = atomic_load_explicit(&tp->n_graph, memory_order_relaxed);
        gen2_started = ggml_time_us();
        gen2_topology = gen2_cpu_profile_topology(cgraph);
    }
'''

AFTER_KERNEL = br'''
        const int64_t gen2_after_kernel = gen2_profile ? ggml_time_us() : 0;
        if (gen2_profile) {
            gen2_ops[node->op].roots++;
            gen2_ops[node->op].fused += n_fused;
            gen2_ops[node->op].kernel_us += gen2_after_kernel - gen2_before_kernel;
            gen2_roots++;
            gen2_fused += n_fused;
            gen2_execution = gen2_cpu_profile_mix(gen2_execution, (uint64_t) gen2_root_index);
            gen2_execution = gen2_cpu_profile_mix(gen2_execution, (uint64_t) node->op);
            gen2_execution = gen2_cpu_profile_mix(gen2_execution, (uint64_t) n_fused);
        }
'''

EMIT = br'''
    if (gen2_profile) {
        const int64_t gen2_finished = ggml_time_us();
        fprintf(stderr, "[DEBUG-cpu-ops-v1] graph pool=%p generation=%d thread=%d threads=%d nodes=%d roots=%d fused=%d skipped=%d topology=%016" PRIx64 " execution=%016" PRIx64 " elapsed_us=%" PRId64 " final_wait_us=%" PRId64 "\n",
                (void *) tp, gen2_generation, state->ith, params.nth, gen2_nodes,
                gen2_roots, gen2_fused, gen2_skipped, gen2_topology, gen2_execution,
                gen2_finished - gen2_started, gen2_finished - gen2_before_final);
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            const struct gen2_cpu_op_timing * value = &gen2_ops[op];
            if (value->roots != 0) {
                fprintf(stderr, "[DEBUG-cpu-ops-v1] op pool=%p generation=%d thread=%d id=%d name=%s roots=%d fused=%d kernel_us=%" PRId64 " barrier_us=%" PRId64 "\n",
                        (void *) tp, gen2_generation, state->ith, op,
                        ggml_op_name((enum ggml_op) op), value->roots, value->fused,
                        value->kernel_us, value->barrier_us);
            }
        }
    }
'''


def replace_once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('one exact pinned CPU timing insertion site required')
    return raw.replace(old, new)


def transform(raw):
    """Require the complete canonical nineteen-patch file, then add timers only."""
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('exact canonical CPU source required for diagnostic profiling')
    start = raw.index(b'static thread_ret_t ggml_graph_compute_thread(void * data) {')
    end = raw.index(b'\n#ifndef GGML_USE_OPENMP\n', start)
    body = raw[start:end]
    site = b'    const struct ggml_cplan  * cplan  = tp->cplan;\n'
    body = replace_once(body, site, site + INITIALIZE)
    for site in (b'        if (ggml_op_is_empty(node->op)) {\n',
                 b'        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {\n'):
        body = replace_once(body, site, site + b'            if (gen2_profile) { gen2_skipped++; }\n')
    site = b'        // TODO: move fused-op detection into ggml_graph_plan'
    body = replace_once(body, site,
        b'        const int gen2_root_index = node_n;\n'
        b'        const int64_t gen2_before_kernel = gen2_profile ? ggml_time_us() : 0;\n\n' + site)
    site = b'        if (state->ith == 0 && cplan->abort_callback &&\n'
    body = replace_once(body, site, AFTER_KERNEL + b'\n' + site)
    site = b'''        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
'''
    body = replace_once(body, site,
        b'        const int64_t gen2_before_outer = gen2_profile ? ggml_time_us() : 0;\n' + site + br'''        if (gen2_profile) {
            gen2_ops[node->op].barrier_us += ggml_time_us() - gen2_before_outer;
        }
''')
    site = b'    ggml_barrier(state->threadpool);\n\n#ifdef GGML_USE_CPU_RISCV64_SPACEMIT'
    body = replace_once(body, site,
        b'    const int64_t gen2_before_final = gen2_profile ? ggml_time_us() : 0;\n'
        b'    ggml_barrier(state->threadpool);\n' + EMIT + b'\n#ifdef GGML_USE_CPU_RISCV64_SPACEMIT')
    return raw[:start] + HELPERS + body + raw[end:]

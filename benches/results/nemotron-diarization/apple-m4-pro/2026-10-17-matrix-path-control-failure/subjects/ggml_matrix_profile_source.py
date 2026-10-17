"""Exact-source observer of existing matrix dispatch; no new arithmetic/barriers."""
from ggml_cpu_profile_source import transform as cpu_transform, replace_once, PATH, ORIGINAL_SHA256

ENVIRONMENT='GEN2_DIAR_MATRIX_PROFILE'
PREFIX='[DEBUG-matrix-paths-v1]'
MAX_MATRICES=256

HELPERS=br'''
// Diagnostic matrix-path state belongs only to the executing worker/graph.
struct gen2_matrix_stage { int calls, ok, failed; int64_t us; };
struct gen2_matrix_record {
    int node, types[3], dot, route, converted;
    const char * names[3];
    int64_t ne[3][4];
    uint64_t nb[3][4];
    struct gen2_matrix_stage stage[2];
    int64_t conversion_us, inner_wait_us, chunk_us, kernel_us;
    uint64_t chunks;
};
static _Thread_local struct gen2_matrix_record * gen2_matrix_current;

#if GGML_USE_LLAMAFILE
static bool gen2_matrix_sgemm(int stage, const struct ggml_compute_params * params,
        int64_t m, int64_t n, int64_t k, const void * a, int64_t lda,
        const void * b, int64_t ldb, void * c, int64_t ldc, int at, int bt, int ct) {
    struct gen2_matrix_record * record = gen2_matrix_current;
    const int64_t start = record ? ggml_time_us() : 0;
    const bool result = llamafile_sgemm(params, m, n, k, a, lda, b, ldb, c, ldc, at, bt, ct);
    if (record) {
        struct gen2_matrix_stage * value = &record->stage[stage];
        value->calls++;
        value->ok += result;
        value->failed += !result;
        value->us += ggml_time_us() - start;
    }
    return result;
}
#endif

'''

INITIALIZE=br'''
    struct gen2_matrix_record gen2_matrices[256];
    int gen2_matrix_count = 0;
    bool gen2_matrix_overflow = false;
    gen2_matrix_current = NULL;
'''

BEGIN=br'''
        gen2_matrix_current = NULL;
        if (gen2_profile && node->op == GGML_OP_MUL_MAT) {
            if (gen2_matrix_count == 256) {
                gen2_matrix_overflow = true;
            } else {
                struct gen2_matrix_record * value = &gen2_matrices[gen2_matrix_count++];
                memset(value, 0, sizeof(*value));
                value->node = gen2_root_index;
                value->dot = -1;
                const struct ggml_tensor * tensors[3] = { node->src[0], node->src[1], node };
                for (int t = 0; t < 3; ++t) {
                    value->types[t] = tensors[t]->type;
                    value->names[t] = ggml_type_name(tensors[t]->type);
                    for (int d = 0; d < 4; ++d) {
                        value->ne[t][d] = tensors[t]->ne[d];
                        value->nb[t][d] = tensors[t]->nb[d];
                    }
                }
                gen2_matrix_current = value;
            }
        }
'''

FINISH=br'''
        if (gen2_matrix_current) {
            if (gen2_matrix_current->route == 0 && n_fused > 0) gen2_matrix_current->route = 7;
            gen2_matrix_current->kernel_us = gen2_after_kernel - gen2_before_kernel;
            gen2_matrix_current = NULL;
        }
'''

EMIT=br'''
    if (gen2_profile) {
        if (gen2_matrix_overflow) fprintf(stderr, "[DEBUG-matrix-paths-v1] overflow\n");
        for (int i = 0; i < gen2_matrix_count; ++i) {
            const struct gen2_matrix_record * v = &gen2_matrices[i];
            char shape[3][128], stride[3][128];
            for (int t = 0; t < 3; ++t) {
                snprintf(shape[t], sizeof(shape[t]), "%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64,
                         v->ne[t][0], v->ne[t][1], v->ne[t][2], v->ne[t][3]);
                snprintf(stride[t], sizeof(stride[t]), "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
                         v->nb[t][0], v->nb[t][1], v->nb[t][2], v->nb[t][3]);
            }
            fprintf(stderr, "[DEBUG-matrix-paths-v1] matrix pool=%p generation=%d thread=%d node=%d types=%d,%d,%d names=%s,%s,%s a=%s b=%s c=%s sa=%s sb=%s sc=%s dot=%d route=%d first=%d,%d,%d,%" PRId64 " second=%d,%d,%d,%" PRId64 " conversion=%d,%" PRId64 " inner_wait_us=%" PRId64 " chunks=%" PRIu64 ",%" PRId64 " kernel_us=%" PRId64 "\n",
                    (void *) tp, gen2_generation, state->ith, v->node,
                    v->types[0], v->types[1], v->types[2], v->names[0], v->names[1], v->names[2],
                    shape[0], shape[1], shape[2], stride[0], stride[1], stride[2], v->dot, v->route,
                    v->stage[0].calls, v->stage[0].ok, v->stage[0].failed, v->stage[0].us,
                    v->stage[1].calls, v->stage[1].ok, v->stage[1].failed, v->stage[1].us,
                    v->converted, v->conversion_us, v->inner_wait_us, v->chunks, v->chunk_us, v->kernel_us);
        }
    }
'''


def transform(raw):
    result=cpu_transform(raw)
    result=replace_once(result,b'getenv("GEN2_DIAR_CPU_PROFILE")',b'getenv("GEN2_DIAR_MATRIX_PROFILE")')
    start=result.index(b'void ggml_compute_forward_mul_mat(\n')
    end=result.index(b'\n// ggml_compute_forward_mul_mat_id',start)
    body=result[start:end]
    body=replace_once(body,b'        ggml_compute_forward_fwht(params, dst);',
        b'        if (gen2_matrix_current) gen2_matrix_current->route = 1;\n        ggml_compute_forward_fwht(params, dst);')
    site=b'    ggml_from_float_t        const from_float'
    body=replace_once(body,site,b'    if (gen2_matrix_current) gen2_matrix_current->dot = vec_dot_type;\n'+site)
    if body.count(b'if (!llamafile_sgemm(params,')!=2:raise ValueError('two original SGEMM stages required')
    body=body.replace(b'if (!llamafile_sgemm(params,',b'if (!gen2_matrix_sgemm(0, params,',1)
    body=body.replace(b'if (!llamafile_sgemm(params,',b'if (!gen2_matrix_sgemm(1, params,',1)
    body=replace_once(body,b'        return;\n    }\nUseGgmlGemm1:;',b'        if (gen2_matrix_current) gen2_matrix_current->route = 2;\n        return;\n    }\nUseGgmlGemm1:;')
    body=replace_once(body,b'        return;\n    }\nUseGgmlGemm2:;',b'        if (gen2_matrix_current) gen2_matrix_current->route = 4;\n        return;\n    }\nUseGgmlGemm2:;')
    site=b'    if (src1->type != vec_dot_type) {\n        char * wdata'
    body=replace_once(body,site,b'    int64_t gen2_conversion_start = 0;\n'+site.replace(b'        char * wdata',b'        if (gen2_matrix_current) { gen2_matrix_current->converted = 1; gen2_conversion_start = ggml_time_us(); }\n        char * wdata'))
    site=b'    if (ith == 0) {\n'
    body=replace_once(body,site,b'    if (gen2_matrix_current && gen2_matrix_current->converted) gen2_matrix_current->conversion_us = ggml_time_us() - gen2_conversion_start;\n\n'+site)
    site=b'    ggml_barrier(params->threadpool);\n'
    body=replace_once(body,site,b'    const int64_t gen2_wait_start = gen2_matrix_current ? ggml_time_us() : 0;\n'+site+b'    if (gen2_matrix_current) gen2_matrix_current->inner_wait_us = ggml_time_us() - gen2_wait_start;\n')
    site=b'        ggml_compute_forward_mul_mat_iqp(params, dst);'
    body=replace_once(body,site,b'        if (gen2_matrix_current) gen2_matrix_current->route = 3;\n'+site)
    site=b'    // This is the size of the first dimension of the result'
    body=replace_once(body,site,b'    if (gen2_matrix_current) gen2_matrix_current->route = 5;\n\n'+site)
    site=b'        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);'
    body=replace_once(body,site,b'        const int64_t gen2_chunk_start = gen2_matrix_current ? ggml_time_us() : 0;\n'+site+b'\n        if (gen2_matrix_current) { gen2_matrix_current->chunks++; gen2_matrix_current->chunk_us += ggml_time_us() - gen2_chunk_start; }')
    result=result[:start]+HELPERS+body+result[end:]
    site=b'    if (ggml_cpu_extra_compute_forward(params, tensor)) {\n'
    result=replace_once(result,site,br'''    struct gen2_matrix_record * gen2_saved_matrix = gen2_matrix_current;
    gen2_matrix_current = NULL;
    const bool gen2_extra = ggml_cpu_extra_compute_forward(params, tensor);
    gen2_matrix_current = gen2_saved_matrix;
    if (gen2_extra) {
        if (tensor->op == GGML_OP_MUL_MAT && gen2_matrix_current) gen2_matrix_current->route = 6;
''')
    site=b'        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);\n'
    result=replace_once(result,site,br'''        struct gen2_matrix_record * gen2_saved_matrix = gen2_matrix_current;
        gen2_matrix_current = NULL;
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        gen2_matrix_current = gen2_saved_matrix;
''')
    site=b'    struct gen2_cpu_op_timing gen2_ops[GGML_OP_COUNT];\n'
    result=replace_once(result,site,site+INITIALIZE)
    site=b'        const int gen2_root_index = node_n;\n'
    result=replace_once(result,site,site+BEGIN)
    site=b'        const int64_t gen2_after_kernel = gen2_profile ? ggml_time_us() : 0;\n'
    result=replace_once(result,site,site+FINISH)
    site=b'\n#ifdef GGML_USE_CPU_RISCV64_SPACEMIT\n    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);'
    result=replace_once(result,site,b'\n'+EMIT+site)
    return result

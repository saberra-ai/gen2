"""Separate diagnostic candidate: batch only the existing post-barrier matrix rows."""
from ggml_matrix_profile_source import EMIT as ORIGINAL_EMIT, transform as original_transform
from ggml_cpu_profile_source import PATH, ORIGINAL_SHA256, replace_once


# Keep the v1 wire schema, capture lifetime, timers and arithmetic unchanged.
# Each row has a checked 2048-byte budget. At most 256 rows are captured by the
# original observer; the temporary heap allocation is therefore at most 512 KiB.
EMIT = replace_once(ORIGINAL_EMIT,
    b'        for (int i = 0; i < gen2_matrix_count; ++i) {', br'''
        const size_t gen2_capacity = (size_t) gen2_matrix_count * 2048;
        char * gen2_buffer = gen2_capacity ? malloc(gen2_capacity) : NULL;
        size_t gen2_used = 0;
        bool gen2_emission_error = gen2_capacity && !gen2_buffer;
        for (int i = 0; i < gen2_matrix_count && !gen2_emission_error; ++i) {''')
EMIT = replace_once(EMIT,
    b'            fprintf(stderr, "[DEBUG-matrix-paths-v1] matrix',
    b'            const int gen2_written = snprintf(gen2_buffer + gen2_used, gen2_capacity - gen2_used, "[DEBUG-matrix-paths-v1] matrix')
EMIT = replace_once(EMIT,
    b'v->chunks, v->chunk_us, v->kernel_us);\n        }\n    }\n', br'''v->chunks, v->chunk_us, v->kernel_us);
            if (gen2_written < 0 || (size_t) gen2_written >= gen2_capacity - gen2_used) {
                gen2_emission_error = true;
            } else {
                gen2_used += (size_t) gen2_written;
            }
        }
        if (!gen2_emission_error && gen2_used && fwrite(gen2_buffer, 1, gen2_used, stderr) != gen2_used) {
            gen2_emission_error = true;
        }
        if (gen2_buffer) free(gen2_buffer);
        if (gen2_emission_error) fprintf(stderr, "[DEBUG-matrix-paths-v1] emission-error\n");
    }
''')


def transform(raw):
    """Require the same canonical source; change only the final emission block."""
    return replace_once(original_transform(raw), ORIGINAL_EMIT, EMIT)

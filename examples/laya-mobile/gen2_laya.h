#ifndef GEN2_LAYA_H
#define GEN2_LAYA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Every response is owned JSON {ok, value|error}; release it exactly once. */
char *gen2_laya_open(const char *config_json);
char *gen2_laya_decide(uint64_t handle, const char *request_json);
/* operation: describe | decide | batch | long; see README for JSON options. */
char *gen2_laya_invoke(uint64_t handle, const char *invocation_json);
char *gen2_laya_suspend(uint64_t handle);
char *gen2_laya_resume(uint64_t handle);
char *gen2_laya_close(uint64_t handle);
void gen2_laya_free(char *response);
#ifdef __cplusplus
}
#endif
#endif

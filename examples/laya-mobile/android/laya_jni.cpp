#include <jni.h>
#include <cstring>
#include <vector>
#include "../gen2_laya.h"

static jbyteArray consume(JNIEnv *env, char *response) {
    if (!response) return nullptr;
    auto length = static_cast<jsize>(std::strlen(response));
    auto result = env->NewByteArray(length);
    if (result) env->SetByteArrayRegion(result, 0, length, reinterpret_cast<const jbyte *>(response));
    gen2_laya_free(response);
    return result;
}
static std::vector<char> utf8(JNIEnv *env, jbyteArray input) {
    auto size = input ? env->GetArrayLength(input) : 0;
    std::vector<char> bytes(static_cast<size_t>(size) + 1, 0);
    if (input) env->GetByteArrayRegion(input, 0, size, reinterpret_cast<jbyte *>(bytes.data()));
    return bytes;
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_example_gen2_LayaNative_openBytes(JNIEnv *env, jclass, jbyteArray config) {
    auto bytes = utf8(env, config);
    if (env->ExceptionCheck()) return nullptr;
    return consume(env, gen2_laya_open(bytes.data()));
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_example_gen2_LayaNative_decideBytes(JNIEnv *env, jclass, jlong handle, jbyteArray request) {
    auto bytes = utf8(env, request);
    if (env->ExceptionCheck()) return nullptr;
    return consume(env, gen2_laya_decide(static_cast<uint64_t>(handle), bytes.data()));
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_example_gen2_LayaNative_suspendBytes(JNIEnv *env, jclass, jlong handle) {
    return consume(env, gen2_laya_suspend(static_cast<uint64_t>(handle)));
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_example_gen2_LayaNative_invokeBytes(JNIEnv *env, jclass, jlong handle, jbyteArray invocation) {
    auto bytes = utf8(env, invocation);
    if (env->ExceptionCheck()) return nullptr;
    return consume(env, gen2_laya_invoke(static_cast<uint64_t>(handle), bytes.data()));
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_example_gen2_LayaNative_resumeBytes(JNIEnv *env, jclass, jlong handle) {
    return consume(env, gen2_laya_resume(static_cast<uint64_t>(handle)));
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_example_gen2_LayaNative_closeBytes(JNIEnv *env, jclass, jlong handle) {
    return consume(env, gen2_laya_close(static_cast<uint64_t>(handle)));
}

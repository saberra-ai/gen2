package example.gen2;

import java.nio.charset.StandardCharsets;

/** Run open/decide/resume/close on an executor, never the main UI thread. */
public final class LayaNative {
    static { System.loadLibrary("gen2_laya_jni"); }
    private LayaNative() {}
    private static native byte[] openBytes(byte[] config);
    private static native byte[] decideBytes(long handle, byte[] request);
    private static native byte[] invokeBytes(long handle, byte[] invocation);
    private static native byte[] suspendBytes(long handle);
    private static native byte[] resumeBytes(long handle);
    private static native byte[] closeBytes(long handle);
    private static String text(byte[] bytes) { return new String(bytes, StandardCharsets.UTF_8); }
    public static String open(String config) { return text(openBytes(config.getBytes(StandardCharsets.UTF_8))); }
    public static String decide(long handle, String request) { return text(decideBytes(handle, request.getBytes(StandardCharsets.UTF_8))); }
    public static String invoke(long handle, String invocation) { return text(invokeBytes(handle, invocation.getBytes(StandardCharsets.UTF_8))); }
    public static String suspend(long handle) { return text(suspendBytes(handle)); }
    public static String resume(long handle) { return text(resumeBytes(handle)); }
    public static String close(long handle) { return text(closeBytes(handle)); }
}

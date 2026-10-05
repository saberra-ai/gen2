package example.gen2;

import android.app.Activity;
import android.os.Bundle;
import android.os.Build;
import android.os.SystemClock;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/** Offline host harness. Real checkpoint qualification requires device evidence. */
public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final AtomicLong handle = new AtomicLong();
    private volatile boolean foreground;
    private volatile boolean destroyed;
    private TextView status;
    private Button run;
    private Spinner families;
    private boolean ciSmokePending;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (20 * getResources().getDisplayMetrics().density);
        column.setPadding(pad, pad, pad, pad);
        TextView title = new TextView(this);
        title.setText(R.string.title);
        title.setTextSize(24);
        column.addView(title);
        TextView instructions = new TextView(this);
        instructions.setText(R.string.instructions);
        column.addView(instructions);
        families = new Spinner(this);
        families.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
            new String[]{"smoke", "english", "multilingual", "typed-decisions"}));
        column.addView(families);
        run = new Button(this);
        run.setText(R.string.run);
        column.addView(run);
        status = new TextView(this);
        status.setTextIsSelectable(true);
        status.setText(R.string.ready);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(status);
        column.addView(scroll);
        setContentView(column);
        run.setOnClickListener(v -> start((String) families.getSelectedItem()));
        ciSmokePending = getIntent().getBooleanExtra("ci_smoke", false);
    }

    private void start(String family) {
        run.setEnabled(false);
        families.setEnabled(false);
        status.setText(getString(R.string.running, family));
        worker.execute(() -> exercise(family));
    }

    private static JSONObject value(String reply) throws Exception {
        JSONObject response = new JSONObject(reply);
        if (!response.getBoolean("ok")) throw new IOException(response.optString("error"));
        return response.optJSONObject("value");
    }

    private void exercise(String family) {
        JSONObject report = new JSONObject();
        try {
            report.put("schema", "gen2-laya-android-smoke/v1");
            report.put("family", family);
            report.put("run_id", getIntent().getStringExtra("ci_run_id"));
            report.put("synthetic", family.equals("smoke"));
            report.put("device", Build.MANUFACTURER + " " + Build.MODEL);
            report.put("android_sdk", Build.VERSION.SDK_INT);
            report.put("abi", Build.SUPPORTED_ABIS[0]);
            File bundle = new File(getFilesDir(), "bundles/" + family);
            if (family.equals("smoke")) copyAssetTree("smoke", bundle);
            JSONObject config = new JSONObject().put("bundle", bundle.getAbsolutePath())
                .put("native_library", JSONObject.NULL).put("resident_budget_mb", family.equals("smoke") ? 16 : 4096)
                .put("queue_capacity", 2).put("intra_threads", 1).put("max_input_bytes", 1048576);
            long started = SystemClock.elapsedRealtimeNanos();
            long id = value(LayaNative.open(config.toString())).getLong("handle");
            handle.set(id);
            report.put("load_millis", (SystemClock.elapsedRealtimeNanos() - started) / 1e6);
            requireForeground();
            report.put("description", value(LayaNative.invoke(id, "{\"operation\":\"describe\"}")));
            String request = "{\"state\":{\"kind\":\"text\",\"value\":\"Please refund the duplicate payment. Café 😀\"},\"questions\":[[\"refund\",{\"type\":\"yes_no\",\"instructions\":\"Is a refund requested?\",\"false_label\":\"false\",\"true_label\":\"true\",\"false_description\":null,\"true_description\":null}]]}";
            started = SystemClock.elapsedRealtimeNanos();
            JSONObject first = value(LayaNative.decide(id, request));
            report.put("inference_millis", (SystemClock.elapsedRealtimeNanos() - started) / 1e6);
            report.put("first", first);
            JSONObject batchCall = new JSONObject().put("operation", "batch")
                .put("requests", new JSONArray().put(new JSONObject(request)).put(new JSONObject(request)))
                .put("options", new JSONObject().put("timeout_ms", 60000));
            JSONObject batchReply = new JSONObject(LayaNative.invoke(id, batchCall.toString()));
            if (!batchReply.getBoolean("ok")) throw new IOException(batchReply.optString("error"));
            JSONArray batch = batchReply.getJSONArray("value");
            if (batch.length() != 2) throw new IOException("Batch result count mismatch");
            for (int i = 0; i < batch.length(); i++) {
                if (!equivalent(first.getJSONArray("answers"), batch.getJSONObject(i).getJSONArray("answers")))
                    throw new IOException("Batch changed answers");
            }
            report.put("batch", batch);
            requireForeground();
            StringBuilder longText = new StringBuilder();
            for (int i = 0; i < 40; i++) longText.append("a ");
            JSONObject longRequest = new JSONObject(request).put("state",
                new JSONObject().put("kind", "text").put("value", longText.toString()));
            JSONObject scanCall = new JSONObject().put("operation", "long").put("request", longRequest)
                .put("scan", new JSONObject().put("window_tokens", 16).put("stride_tokens", 8).put("max_windows", 64));
            JSONObject scan = value(LayaNative.invoke(id, scanCall.toString()));
            if (scan.getJSONArray("windows").length() < 2) throw new IOException("Expected multiple scan windows");
            report.put("long_scan", scan);
            JSONObject expiredCall = new JSONObject().put("operation", "decide").put("request", new JSONObject(request))
                .put("options", new JSONObject().put("timeout_ms", 0));
            JSONObject expired = new JSONObject(LayaNative.invoke(id, expiredCall.toString()));
            if (expired.getBoolean("ok") || !expired.optString("error").toLowerCase(java.util.Locale.ROOT).contains("deadline"))
                throw new IOException("Expired deadline was not rejected");
            report.put("deadline_error", expired.getString("error"));
            value(LayaNative.suspend(id));
            JSONObject suspended = new JSONObject(LayaNative.decide(id, request));
            if (suspended.getBoolean("ok")) throw new IOException("Suspended model accepted inference");
            report.put("suspended_error", suspended.getString("error"));
            requireForeground();
            value(LayaNative.resume(id));
            requireForeground();
            JSONObject second = value(LayaNative.decide(id, request));
            // Nondeterministic timing is excluded; every semantic field is retained.
            first.remove("queue_micros"); first.remove("execution_micros");
            second.remove("queue_micros"); second.remove("execution_micros");
            if (!equivalent(first, second)) throw new IOException("Resume changed semantic output");
            value(LayaNative.close(handle.getAndSet(0)));
            JSONObject closed = new JSONObject(LayaNative.decide(id, request));
            if (closed.getBoolean("ok")) throw new IOException("Closed handle accepted inference");
            report.put("closed_error", closed.getString("error"));
            report.put("passed", true);
        } catch (Throwable error) {
            try { report.put("passed", false).put("error", error.toString()); } catch (JSONException ignored) { }
        } finally {
            long id = handle.getAndSet(0);
            if (id != 0) {
                try { value(LayaNative.close(id)); }
                catch (Throwable error) {
                    try { report.put("passed", false).put("cleanup_error", error.toString()); } catch (JSONException ignored) { }
                }
            }
        }
        String text = report.toString();
        try {
            File reports = new File(getFilesDir(), "reports");
            if (!reports.isDirectory() && !reports.mkdirs()) throw new IOException("Cannot create report directory");
            try (FileOutputStream output = new FileOutputStream(new File(reports, family + "-smoke.json"))) {
                output.write(text.getBytes(StandardCharsets.UTF_8));
            }
        } catch (IOException error) { text += "\nReport write failed: " + error; }
        final String displayed = text;
        runOnUiThread(() -> {
            if (!destroyed) {
                status.setText(displayed);
                run.setEnabled(true);
                families.setEnabled(true);
            }
        });
    }

    private void requireForeground() throws IOException {
        if (!foreground || destroyed) throw new IOException("App backgrounded; rerun after returning");
    }

    private void copyAssetTree(String source, File destination) throws IOException {
        String[] children = getAssets().list(source);
        if (children != null && children.length > 0) {
            if (!destination.isDirectory() && !destination.mkdirs()) throw new IOException("Cannot create bundle directory");
            for (String child : children) copyAssetTree(source + "/" + child, new File(destination, child));
        } else {
            try (InputStream input = getAssets().open(source); OutputStream output = new FileOutputStream(destination)) {
                byte[] bytes = new byte[8192];
                int count;
                while ((count = input.read(bytes)) != -1) output.write(bytes, 0, count);
            }
        }
    }

    private static boolean equivalent(Object a, Object b) throws JSONException {
        if (a instanceof JSONObject && b instanceof JSONObject) {
            JSONObject left = (JSONObject) a, right = (JSONObject) b;
            if (left.length() != right.length()) return false;
            java.util.Iterator<String> keys = left.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                if (!right.has(key) || !equivalent(left.get(key), right.get(key))) return false;
            }
            return true;
        }
        if (a instanceof JSONArray && b instanceof JSONArray) {
            JSONArray left = (JSONArray) a, right = (JSONArray) b;
            if (left.length() != right.length()) return false;
            for (int i = 0; i < left.length(); i++) if (!equivalent(left.get(i), right.get(i))) return false;
            return true;
        }
        if (a instanceof Number && b instanceof Number)
            return Math.abs(((Number) a).doubleValue() - ((Number) b).doubleValue()) <= 0.0001;
        return a.equals(b);
    }

    private void suspendActive() {
        long id = handle.get();
        if (id != 0) LayaNative.suspend(id); // Nonblocking; worker retains any active kernel.
    }
    @Override protected void onStart() { super.onStart(); foreground = true; }
    @Override protected void onPostResume() {
        super.onPostResume();
        if (ciSmokePending) { ciSmokePending = false; start("smoke"); }
    }
    @Override protected void onStop() { foreground = false; suspendActive(); super.onStop(); }
    @Override public void onTrimMemory(int level) {
        super.onTrimMemory(level);
        if (level >= TRIM_MEMORY_RUNNING_LOW) suspendActive();
    }
    @Override protected void onDestroy() {
        destroyed = true;
        foreground = false;
        suspendActive();
        worker.shutdown(); // Accepted job owns cleanup; never block the UI thread.
        super.onDestroy();
    }
}

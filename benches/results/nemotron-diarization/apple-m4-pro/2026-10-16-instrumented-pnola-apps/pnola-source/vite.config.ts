import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { watch: { ignored: ["**/src-tauri/**", "**/native/**", "**/.desktop-assets/**"] } },
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  build: {
    rolldownOptions: {
      onwarn(warning, defaultHandler) {
        if (warning.code !== "MODULE_LEVEL_DIRECTIVE") defaultHandler(warning);
      },
    },
  },
});

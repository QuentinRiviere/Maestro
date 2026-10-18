import path from "node:path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The build lands in the FastAPI static directory: `index()` serves
// static/index.html and the assets resolve under /static/assets/.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/static/",
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  build: { outDir: "../model_hub/static", emptyOutDir: true, sourcemap: false },
});

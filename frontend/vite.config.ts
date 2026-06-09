import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev` the UI is served from :5173 while the Python API runs on
// :8000. We proxy /api and /ws (with ws upgrade) so the front-end can use plain
// same-origin paths everywhere; in production FastAPI serves the built dist at
// the same origin, so the very same relative URLs just work.
const API_TARGET = process.env.PIEEG_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
      "/ws": { target: API_TARGET, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});

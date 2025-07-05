import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "localhost",
    port: 5173,
    hmr: {
      port: 5173,
    },
    proxy: {
      '/transcribe': 'http://127.0.0.1:5000',
      '/query': 'http://127.0.0.1:5000',
      '/generate_highlights': 'http://127.0.0.1:5000',
      '/export_reel': 'http://127.0.0.1:5000',
      '/download_reel': 'http://127.0.0.1:5000',
      '/clear_conversation': 'http://127.0.0.1:5000',
      '/library': 'http://127.0.0.1:5000',
      '/delete_job': 'http://127.0.0.1:5000',
    },
  },
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));

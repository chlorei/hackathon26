import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run dev` proxies the API to `dreamkit ui` on :8800; `npm run build` writes into the Python package.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: '../dreamkit/ui_dist', emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8800',
      '/media': 'http://127.0.0.1:8800',
      '/ws': { target: 'ws://127.0.0.1:8800', ws: true },
    },
  },
})

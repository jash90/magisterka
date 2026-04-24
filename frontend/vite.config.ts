import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const apiTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000'
const apiProxy = {
  target: apiTarget,
  changeOrigin: true,
  secure: false,
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/predict': apiProxy,
      '/explain': apiProxy,
      '/chat': apiProxy,
      '/config': apiProxy,
      '/model': apiProxy,
      '/health': apiProxy,
      '/agent': apiProxy,
      '/agent/chat': apiProxy,
    },
  },
})

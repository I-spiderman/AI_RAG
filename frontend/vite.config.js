import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    // 开发代理：前端 /api/* 转发到 FastAPI（后端不跨域）
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // 关键：让 SSE 流式响应逐块透传，不被开发代理缓冲后一次性下发
        // （之前"一次性全弹出"的根因就在这一层——Vite 代理默认不保证流式）
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            // 声明不缓冲、分块直传；X-Accel-Buffering 对 nginx 同样有效，一并带上无害
            proxyRes.headers['cache-control'] = 'no-cache, no-transform'
            proxyRes.headers['x-accel-buffering'] = 'no'
          })
        },
      },
    },
  },
})

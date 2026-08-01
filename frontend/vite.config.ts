import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-time bridge to the gateway: the trace browser calls /internal/*
    // same-origin and Vite forwards to the FastAPI backend — no CORS setup.
    proxy: {
      '/internal': 'http://localhost:8000',
    },
  },
})

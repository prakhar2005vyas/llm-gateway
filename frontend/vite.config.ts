import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-time bridge to the gateway: the trace browser calls /internal/*
    // same-origin and Vite forwards to the FastAPI backend — no CORS setup.
    //
    // The Authorization header is injected HERE (Node.js proxy layer) so the
    // bearer token is NEVER inlined into the browser JavaScript bundle.
    // To override the default dev key, set GATEWAY_API_KEY in your shell
    // before running `vite dev` — it stays server-side only.
    proxy: {
      '/internal': {
        target: 'http://localhost:8000',
        headers: {
          Authorization: `Bearer ${process.env.GATEWAY_API_KEY ?? 'my_secure_local_password'}`,
        },
      },
    },
  },
})

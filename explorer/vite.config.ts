import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Project pages live under /opd/ on github.io; override with VITE_BASE for forks.
const base = process.env.VITE_BASE ?? '/opd/'

export default defineConfig({
  base,
  plugins: [react()],
})

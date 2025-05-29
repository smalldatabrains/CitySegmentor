import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
  },
  server: {
    proxy: process.env.NODE_ENV === 'development' ? {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      }
    } : undefined
  }
});

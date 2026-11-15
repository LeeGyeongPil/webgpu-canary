import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

const demoDir = fileURLToPath(new URL('.', import.meta.url));

// The demo imports the library straight from ../src, so no build step is needed first.
export default defineConfig({
  root: demoDir,
  base: './',
  plugins: [react()],
  build: { outDir: '../demo-dist', emptyOutDir: true },
  server: { host: '127.0.0.1' },
});

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: './',
  plugins: [react()],
  // Geometry fixtures compile complete levels. Bound concurrent compilers so
  // their per-test deadlines do not depend on the host's reported CPU count.
  test: { include: ['tests/**/*.test.ts'], environment: 'node', maxWorkers: 2 },
  build: { target: 'es2022' },
});

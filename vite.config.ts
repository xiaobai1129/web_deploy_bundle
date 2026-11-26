import { cpSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Copies static model/demo/golden asset folders into dist/ on production build
 * without duplicating binary files into a separate public/ directory.
 * In `vite dev`, Vite serves project-root directories (`/weights`, `/demo_videos`, `/golden_vectors`) natively.
 */
function copyBundleAssetsPlugin(): Plugin {
  const assetDirs = ['weights', 'demo_videos', 'golden_vectors'];
  return {
    name: 'copy-bundle-assets',
    closeBundle() {
      const rootDir = __dirname;
      const outDir = resolve(rootDir, 'dist');
      if (!existsSync(outDir)) return;
      for (const dir of assetDirs) {
        const src = resolve(rootDir, dir);
        const dst = resolve(outDir, dir);
        if (existsSync(src)) {
          cpSync(src, dst, { recursive: true });
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), copyBundleAssetsPlugin()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
  },
  assetsInclude: ['**/*.wgsl'],
});

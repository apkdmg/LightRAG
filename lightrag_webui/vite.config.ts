import { defineConfig } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// WebUI prefix - must match the value in src/lib/constants.ts
const webuiPrefix = '/webui/'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  // base: import.meta.env.VITE_BASE_URL || '/webui/',
  base: webuiPrefix,
  build: {
    outDir: path.resolve(__dirname, '../lightrag/api/webui'),
    emptyOutDir: true,
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        // Manual chunking strategy
        manualChunks: {
          // Group React-related libraries into one chunk
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          // Group graph visualization libraries into one chunk
          'graph-vendor': ['sigma', 'graphology', '@react-sigma/core'],
          // Group UI component libraries into one chunk
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-popover', '@radix-ui/react-select', '@radix-ui/react-tabs'],
          // Group utility libraries into one chunk
          'utils-vendor': ['axios', 'i18next', 'zustand', 'clsx', 'tailwind-merge'],
          // Separate feature modules
          'feature-graph': ['./src/features/GraphViewer'],
          'feature-documents': ['./src/features/DocumentManager'],
          'feature-retrieval': ['./src/features/RetrievalTesting'],

          // Mermaid-related modules
          'mermaid-vendor': ['mermaid'],

          // Markdown-related modules
          'markdown-vendor': [
            'react-markdown',
            'rehype-react',
            'remark-gfm',
            'remark-math',
            'react-syntax-highlighter'
          ]
        },
        // Ensure consistent chunk naming format
        chunkFileNames: 'assets/[name]-[hash].js',
        // Entry file naming format
        entryFileNames: 'assets/[name]-[hash].js',
        // Asset file naming format
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  },
  server: {
    // Proxy configuration for development mode
    // Set VITE_API_PROXY=true and VITE_API_ENDPOINTS in .env to enable
    proxy: process.env.VITE_API_PROXY === 'true' && process.env.VITE_API_ENDPOINTS ?
      Object.fromEntries(
        process.env.VITE_API_ENDPOINTS.split(',').map(endpoint => [
          endpoint,
          {
            target: process.env.VITE_BACKEND_URL || 'http://localhost:9621',
            changeOrigin: true,
            rewrite: endpoint === '/api' ?
              (p) => p.replace(/^\/api/, '') :
              endpoint === '/docs' || endpoint === '/openapi.json' ?
                (p) => p : undefined
          }
        ])
      ) : {}
  }
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// During `vite dev` we serve ../data_tools/landmarks/*.json at /landmarks/*.json
// so local development hits landmark files exactly the way production hits R2:
//   fetch(`${VITE_LANDMARKS_BASE_URL}/${name}.json`)
// In production VITE_LANDMARKS_BASE_URL points at the public R2 URL (set in
// Vercel), so this middleware is dev-only. Files in frontend/public/landmarks/
// (the 4 always-bundled fallbacks) take precedence because Vite serves /public
// on the same path before this middleware runs.
const DATA_TOOLS_LANDMARKS = path.resolve(__dirname, '..', 'data_tools', 'landmarks')

function devLandmarksMiddleware() {
  return {
    name: 'serve-data-tools-landmarks',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/landmarks', (req, res, next) => {
        const urlPath = decodeURIComponent((req.url || '').split('?')[0])
        // Only allow simple filenames -- no path traversal.
        if (!/^\/[A-Za-z0-9._-]+\.json$/.test(urlPath)) return next()
        const filePath = path.join(DATA_TOOLS_LANDMARKS, urlPath.slice(1))
        fs.stat(filePath, (err, stat) => {
          if (err || !stat.isFile()) return next() // fall through to public/landmarks
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          fs.createReadStream(filePath).pipe(res)
        })
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), basicSsl(), devLandmarksMiddleware()],
  server: {
    proxy: {
      '/api/llama': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/llama/, '')
      }
    }
  }
})

import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

let server
let App
const previousWindow = globalThis.window
const previousStorage = globalThis.localStorage

before(async () => {
  const values = new Map()
  globalThis.window = {
    location: { search: '?desktop=orb&wakeWordEnabled=false', href: 'http://localhost/' },
  }
  globalThis.localStorage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  }
  server = await createServer({
    root: fileURLToPath(new URL('../', import.meta.url)),
    configFile: false,
    esbuild: { jsx: 'automatic' },
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { middlewareMode: true, watch: null, hmr: false },
  })
  App = (await server.ssrLoadModule('/src/App.jsx')).default
})

after(async () => {
  await server?.close()
  if (previousWindow === undefined) delete globalThis.window
  else globalThis.window = previousWindow
  if (previousStorage === undefined) delete globalThis.localStorage
  else globalThis.localStorage = previousStorage
})

test('desktop renders a visible sleeping orb with microphone activation disabled on startup', () => {
  const html = renderToStaticMarkup(createElement(App))
  assert.match(html, /class="desktop-orb-stage sleeping/)
  assert.match(html, /你好，千问/)
  assert.match(html, /<button[^>]*disabled=""[^>]*aria-label="[^"]*你好，千问/)
  assert.doesNotMatch(html, /desktop-orb-stage hidden/)
})

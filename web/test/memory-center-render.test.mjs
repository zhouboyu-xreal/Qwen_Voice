import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

let server
let MemoryCenter
let MemoryRecordDetail

before(async () => {
  server = await createServer({
    root: fileURLToPath(new URL('../', import.meta.url)), configFile: false,
    esbuild: { jsx: 'automatic' }, optimizeDeps: { noDiscovery: true, include: [] },
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  })
  const { setRuntimeLanguage } = await server.ssrLoadModule('/src/i18n.js')
  setRuntimeLanguage('zh-CN')
  const module = await server.ssrLoadModule('/src/memory/MemoryCenter.jsx')
  MemoryCenter = module.default
  MemoryRecordDetail = module.MemoryRecordDetail
})
after(async () => { await server?.close() })

test('memory center renders independently without browser voice globals', () => {
  const html = renderToStaticMarkup(createElement(MemoryCenter))
  assert.match(html, /我的记忆/)
  assert.match(html, /查看记忆不会激活语音/)
  assert.match(html, /意图与安排/)
  assert.match(html, /我的状态/)
  assert.doesNotMatch(html, /开启麦克风/)
})

test('details escape extracted content and distinguish missing evidence from original excerpts', () => {
  const html = renderToStaticMarkup(createElement(MemoryRecordDetail, {
    data: {
      item: { id: 1, state_type: 'situational_state', state_text: '<script>alert(1)</script>', status: 'active', valid_to: '2020-01-01', confidence: .8 },
      evidence: { total: 2, items: [
        { evidence_type: 'fact', evidence_id: 1, resolved: true, text: '提炼的内容', source_segments: [{ id: 1, text: '原始话语', started_at: '2026-11-01' }] },
        { evidence_type: 'user_claim', evidence_id: 2, resolved: false },
      ] },
      history: { total: 1, items: [{ id: 1, operation: 'revise', previous_payload: { state_text: '旧状态' }, new_payload: { state_text: '新状态' } }] },
    },
  }))
  assert.doesNotMatch(html, /<script>/)
  assert.match(html, /&lt;script&gt;/)
  assert.match(html, /已过有效期/)
  assert.match(html, /来源记录已不可用/)
  assert.match(html, /原始话语/)
  assert.match(html, /旧状态 → 新状态/)
})

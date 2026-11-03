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
  assert.match(html, /我的活动/)
  assert.doesNotMatch(html, /开启麦克风/)
})

test('activity details separate observation windows, related facts and source context', () => {
  const html = renderToStaticMarkup(createElement(MemoryRecordDetail, { data: {
    item: { id: 1, activity_text: '讨论旅行 <script>test</script>', user_role: 'contextual', confidence: .8,
      started_at: '2026-11-03T10:00:00+08:00', ended_at: '2026-11-03T03:00:00Z', created_at: '2026-11-04T00:00:00Z' },
    evidence: { total: 0, items: [] },
    sources: { total: 7, pageSize: 6, items: [
      { id: 1, resolved: true, text: '明天准备去旅行', started_at: '2026-11-03T10:00:00+08:00' },
      { id: 2, resolved: false },
    ] },
  } }))
  for (const text of ['情境接触', '观察时间窗口', '提炼保存时间', '关联事实', '来源窗口',
    '不代表精确活动时长', '明天准备去旅行', '来源记录已不可用', '来源窗口分页']) assert.ok(html.includes(text), text)
  assert.doesNotMatch(html, /<script>|变更记录|存储状态|有效期至/)
  assert.match(html, /&lt;script&gt;/)
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

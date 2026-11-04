import assert from 'node:assert/strict'
import { once } from 'node:events'
import { createHash } from 'node:crypto'
import test from 'node:test'
import express from 'express'
import { feishuDocument, registerFeishuImportRoutes } from '../src/conversation/memory/feishu-import-api.mjs'
import { FrontendMemoryRuntime } from '../src/conversation/memory/runtime.mjs'
import { AgentMemoryProvider } from '../src/conversation/memory/providers/agent-memory/provider.mjs'
const file = { filename: '会议.md', content: '会议文字', ownerName: '本人' }

test('document validation accepts a single bounded Markdown and never forwards paths or client identity', () => {
  assert.deepEqual(feishuDocument({ ...file, ownerId: 'forged', path: '/other/file' }, true), file)
  for (const body of [{ ...file, filename: '../会议.md' }, { ...file, filename: 'x.txt' },
    { ...file, content: '中'.repeat(200000) }, { ...file, content: '\0' }, { ...file, ownerName: '' }]) {
    assert.equal(feishuDocument(body, true), null)
  }
})

test('routes bind all operations to authenticated owner and sanitize errors', async t => {
  const app = express(); app.use(express.json())
  app.use((req, _res, next) => { req.identity = { ownerId: 'trusted-owner' }; next() })
  let result = { ok: true, id: 'a'.repeat(64), status: 'queued' }
  let fail = false
  const calls = []
  registerFeishuImportRoutes(app, { memory: { feishuMeeting: async (...args) => {
    calls.push(args); if (fail) throw new Error('secret /private/database'); return result
  } } })
  const server = app.listen(0, '127.0.0.1');t.after(() => server.close());await once(server, 'listening')
  const base = `http://127.0.0.1:${server.address().port}/api/pwm/imports/feishu`
  const post = (suffix, body) => fetch(base + suffix, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  assert.equal((await post('/preview', { ...file, ownerId: 'forged' })).status, 200)
  assert.deepEqual(calls[0], ['trusted-owner', 'preview', { filename: file.filename, content: file.content }])
  assert.equal((await post('', file)).status, 202)
  assert.equal((await fetch(base + '/' + 'a'.repeat(64) + '?ownerId=forged')).status, 200)
  assert.deepEqual(calls.at(-1), ['trusted-owner', 'status', { id: 'a'.repeat(64) }])
  assert.equal((await fetch(base + '/invalid')).status, 400)
  for (const [code, status] of [['invalid_feishu_document', 400], ['feishu_import_busy', 409], ['feishu_import_not_found', 404]]) {
    result = { ok: false, code }; assert.equal((await post('', file)).status, status)
  }
  result = { supported: false };assert.equal((await post('', file)).status, 501)
  fail = true
  const response = await post('', file)
  assert.equal(response.status, 503)
  assert.deepEqual(await response.json(), { code: 'feishu_import_unavailable' })
})

test('provider bridge uses the current owner and sidecar rather than starting a test process', async () => {
  const calls = []
  const provider = new AgentMemoryProvider({ sidecar: { start() {}, request: async (...args) => { calls.push(args); return { ok: true } } } })
  const runtime = new FrontendMemoryRuntime({ provider })
  await runtime.feishuMeeting('trusted', 'import', { ...file, ownerId: 'forged' })
  assert.equal(calls[0][0], 'import_feishu_meeting')
  assert.equal(calls[0][1].ownerId, createHash('sha256').update('trusted').digest('hex'))
  assert.equal(calls[0][1].content, file.content)
  const unsupported = new FrontendMemoryRuntime({ provider: { describe: () => ({ protocolVersion: 2, key: 'fixture', label: 'Fixture' }), list: () => [], apply() {} } })
  assert.deepEqual(await unsupported.feishuMeeting('trusted', 'preview', file), { supported: false })
})

import assert from 'node:assert/strict'
import { once } from 'node:events'
import { createHash } from 'node:crypto'
import test from 'node:test'
import express from 'express'
import { registerMemoryResetRoutes } from '../src/conversation/memory/reset-api.mjs'
import { registerMemoryInspectionRoutes } from '../src/conversation/memory/inspection-api.mjs'
import { AgentMemoryProvider } from '../src/conversation/memory/providers/agent-memory/provider.mjs'
import { FrontendMemoryRuntime } from '../src/conversation/memory/runtime.mjs'

test('reset routes require confirmation, bind the authenticated owner and hide backend errors', async t => {
  const app = express(); app.use(express.json())
  app.use((req, _res, next) => { req.identity = { ownerId: 'trusted' }; next() })
  const calls = []; let result = { ok: true, job: null, generation: '0' }, failure = false
  const memory = { resetMemory: async (...args) => {
    calls.push(args); if (failure) throw new Error('secret /private/db'); return result
  } }
  registerMemoryResetRoutes(app, { memory }); registerMemoryInspectionRoutes(app, { memory })
  const server = app.listen(0, '127.0.0.1'); t.after(() => server.close()); await once(server, 'listening')
  const base = `http://127.0.0.1:${server.address().port}/api/pwm/reset`
  const post = body => fetch(base, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  assert.equal((await post({})).status, 400); assert.equal(calls.length, 0)
  assert.equal((await post({ confirmation: 'clear-memory', ownerId: 'forged', path: '/forged' })).status, 202)
  assert.deepEqual(calls.at(-1), ['trusted', 'start'])
  const response = await fetch(base + '?ownerId=forged')
  assert.equal(response.headers.get('cache-control'), 'no-store')
  assert.deepEqual(await response.json(), result); assert.deepEqual(calls.at(-1), ['trusted', 'status'])
  result = { ok: false, code: 'memory_reset_busy' }; assert.equal((await post({ confirmation: 'clear-memory' })).status, 409)
  result = { supported: false }; assert.equal((await fetch(base)).status, 501)
  failure = true; const failed = await fetch(base); assert.equal(failed.status, 503)
  assert.deepEqual(await failed.json(), { code: 'memory_reset_unavailable' })
})

test('a delayed old turn and pending finalization keep the pre-reset token', async () => {
  const calls = []; let generation = 'old'
  let release; const delayed = new Promise(resolve => { release = resolve })
  const provider = new AgentMemoryProvider({ sidecar: {
    request: async (method, params) => {
      calls.push({ method, params })
      if (method === 'memory_generation') return { generation }
      if (method === 'observe' && params.messages[0].id === 'pending') await delayed
      return { ok: true, observed: true }
    }, close() {},
  } })
  const runtime = new FrontendMemoryRuntime({ provider })
  const oldTurn = runtime.memoryGeneration('trusted')
  const pending = provider.observe('trusted', { messages: [{ id: 'pending', role: 'user', content: 'old text' }] }, { sessionId: 'session', memoryGeneration: oldTurn })
  await new Promise(resolve => setImmediate(resolve))
  const finalizing = provider.finalizeSession('trusted', { sessionId: 'session', memoryGeneration: oldTurn })
  generation = 'new'
  await runtime.resetMemory('trusted', 'start')
  await provider.observe('trusted', { messages: [{ id: 'late', role: 'user', content: 'late old text' }] }, { memoryGeneration: oldTurn })
  release(); await pending; await finalizing
  await provider.observe('trusted', { messages: [{ id: 'new', role: 'user', content: 'new text' }] })
  assert.equal(calls.find(call => call.method === 'observe' && call.params.messages[0].id === 'late').params.memoryGeneration, 'old')
  assert.equal(calls.find(call => call.method === 'finalize').params.memoryGeneration, 'old')
  assert.equal(calls.find(call => call.method === 'observe' && call.params.messages[0].id === 'new').params.memoryGeneration, 'new')
  assert.deepEqual(calls.find(call => call.method === 'reset_memory').params, { ownerId: createHash('sha256').update('trusted').digest('hex') })
  assert.deepEqual(await new FrontendMemoryRuntime({ provider: {
    describe: () => ({ protocolVersion: 2, key: 'fixture', label: 'Fixture' }), list: () => [], apply() {},
  } }).resetMemory('trusted', 'start'), { supported: false })
})

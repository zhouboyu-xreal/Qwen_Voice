import assert from 'node:assert/strict'
import test from 'node:test'
import { once } from 'node:events'
import express from 'express'
import { registerMemoryPeriodicTaskRoutes } from '../src/conversation/memory/periodic-task-api.mjs'
import { FrontendMemoryRuntime } from '../src/conversation/memory/runtime.mjs'

test('periodic route uses authenticated identity, whitelists tasks, and sanitizes failures', async t => {
  const calls = []
  let result = { supported: true, queued: true }
  let failure = false
  const app = express()
  app.use(express.json())
  app.use((req, _res, next) => { req.identity = { ownerId: 'authenticated-owner' }; next() })
  registerMemoryPeriodicTaskRoutes(app, { memory: { async triggerPeriodicTask(...args) {
    calls.push(args)
    if (failure) throw new Error('private database /path and credentials')
    return result
  } } })
  const server = app.listen(0, '127.0.0.1')
  t.after(() => server.close())
  await once(server, 'listening')
  const request = body => fetch(`http://127.0.0.1:${server.address().port}/api/memory/periodic-task`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  })
  assert.equal((await request({ taskId: 'unknown' })).status, 400)
  assert.equal(calls.length, 0)
  assert.equal((await request({ taskId: 'user_state_from_claims', ownerId: 'forged-owner', claimIds: [1] })).status, 202)
  assert.deepEqual(calls[0], ['authenticated-owner', 'user_state_from_claims'])
  result = { supported: true, queued: false, reason: 'claim_state_task_pending' }
  assert.equal((await request({ taskId: 'user_state_from_claims' })).status, 200)
  result = { supported: false }
  assert.equal((await request({ taskId: 'user_state_from_claims' })).status, 501)
  failure = true
  const response = await request({ taskId: 'user_state_from_claims' })
  assert.equal(response.status, 503)
  assert.deepEqual(await response.json(), { code: 'periodic_task_unavailable' })
})


test('memory runtime treats periodic signals as an optional advertised capability', async () => {
  const calls = []
  const base = {
    describe: () => ({ protocolVersion: 2, key: 'fixture', label: 'Fixture', capabilities: { periodicTasks: true } }),
    list: () => [], apply: async () => ({ changed: 0, documents: [] }),
  }
  assert.throws(() => new FrontendMemoryRuntime({ provider: base }), /periodicTasks/)
  const provider = { ...base, triggerPeriodicTask: async (...args) => {
    calls.push(args)
    return { supported: true, queued: true }
  } }
  const runtime = new FrontendMemoryRuntime({ provider })
  assert.equal((await runtime.triggerPeriodicTask('owner', 'user_state_from_claims')).queued, true)
  assert.deepEqual(calls, [['owner', 'user_state_from_claims']])
  provider.triggerPeriodicTask = async () => null
  await assert.rejects(() => runtime.triggerPeriodicTask('owner', 'user_state_from_claims'), /must return an object/)
  provider.describe = () => ({ protocolVersion: 1, key: 'legacy', label: 'Legacy' })
  assert.deepEqual(await runtime.triggerPeriodicTask('owner', 'user_state_from_claims'), { supported: false })
})

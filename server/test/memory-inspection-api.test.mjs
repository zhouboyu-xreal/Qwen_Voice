import assert from 'node:assert/strict'
import { once } from 'node:events'
import test from 'node:test'
import express from 'express'
import { memoryInspectionOptions, registerMemoryInspectionRoutes } from '../src/conversation/memory/inspection-api.mjs'

test('validates bounded inspection queries and never forwards client identity or paths', () => {
  const options = memoryInspectionOptions('states', undefined, { ownerId: 'someone-else', path: '/other.db', pageSize: '40' })
  assert.equal(options.pageSize, 40)
  assert.equal(options.ownerId, undefined)
  assert.equal(options.path, undefined)
  for (const [section, id, query] of [
    ['unknown', undefined, {}], ['states', '../2', {}], ['states', undefined, { type: 'goal' }],
    ['intents', undefined, { pageSize: '101' }], ['intents', undefined, { page: ['1'] }],
    ['states', undefined, { scope: 'deleted' }], ['states', undefined, { domain: {} }],
  ]) assert.equal(memoryInspectionOptions(section, id, query), null)
})

test('inspection routes preserve authenticated owner, handle missing records and sanitize failures', async t => {
  const app = express()
  const calls = []
  let result = { supported: true, available: true, items: [], total: 0 }
  let fail = false
  app.use((req, _res, next) => { req.identity = { ownerId: 'authenticated-owner' }; next() })
  registerMemoryInspectionRoutes(app, { memory: {
    inspect: async (owner, options) => {
      calls.push({ owner, options })
      if (fail) throw new Error('private /path/to/database and secret')
      return result
    },
  } })
  const server = app.listen(0, '127.0.0.1')
  t.after(() => server.close())
  await once(server, 'listening')
  const base = `http://127.0.0.1:${server.address().port}`
  let response = await fetch(`${base}/api/pwm/intents?ownerId=other&type=goal`)
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('cache-control'), 'no-store')
  assert.equal(calls[0].owner, 'authenticated-owner')
  assert.equal(calls[0].options.type, 'goal')
  response = await fetch(`${base}/api/pwm/states/99`)
  assert.equal(response.status, 404)
  response = await fetch(`${base}/api/pwm/states?pageSize=1000`)
  assert.equal(response.status, 400)
  result = { supported: false }
  response = await fetch(`${base}/api/pwm/states`)
  assert.equal(response.status, 501)
  fail = true
  response = await fetch(`${base}/api/pwm/intents`)
  assert.equal(response.status, 503)
  assert.deepEqual(await response.json(), { code: 'memory_inspection_unavailable' })
})

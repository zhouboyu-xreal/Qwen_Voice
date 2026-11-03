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
  response = await fetch(`${base}/api/pwm/activities?role=participant&ownerId=other`)
  assert.equal(response.status, 200)
  assert.equal(calls.at(-1).owner, 'authenticated-owner')
  assert.equal(calls.at(-1).options.section, 'activities')
  assert.equal(calls.at(-1).options.role, 'participant')
  response = await fetch(`${base}/api/pwm/activities/99?sourcePage=2`)
  assert.equal(response.status, 404)
  assert.equal(calls.at(-1).options.sourcePage, 2)
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

test('activity queries validate roles and timezone-aware intervals and omit state filters', () => {
  const options = memoryInspectionOptions('activities', undefined, {
    role: 'participant', from: '2026-11-03T08:00:00+08:00', to: '2026-11-04T00:00:00Z',
    scope: 'current', status: 'active', type: 'goal', domain: 'work', ownerId: 'forged',
  })
  assert.equal(options.from, '2026-11-03T00:00:00.000Z')
  assert.equal(options.role, 'participant')
  for (const field of ['type', 'scope', 'status', 'domain', 'ownerId']) assert.equal(options[field], undefined)
  for (const query of [{ role: 'owner' }, { role: ['actor'] }, { sourcePage: '0' },
    { from: '2026-02-30T00:00:00Z' }, { from: '2026-11-03' }, { from: 'invalid' },
    { from: '2026-11-04T00:00:00Z', to: '2026-11-03T00:00:00Z' }]) {
    assert.equal(memoryInspectionOptions('activities', undefined, query), null)
  }
})

import assert from 'node:assert/strict'
import test from 'node:test'
import { createPeriodicTaskController } from '../src/periodic-task-controller.mjs'
import { desktopPwmPeriodicTasks, sendPwmPeriodicSignal } from '../src/pwm-periodic-tasks.mjs'

function harness(tasks = [{ taskId: 'state', intervalMs: 100 }]) {
  let current = 0
  const scheduled = new Map()
  let sequence = 0
  const calls = []
  const errors = []
  const controller = createPeriodicTaskController({
    tasks,
    now: () => current,
    setTimer: (callback, delay) => { const id = ++sequence; scheduled.set(id, { callback, delay }); return id },
    clearTimer: id => scheduled.delete(id),
    onTrigger: fields => { calls.push(fields); return { queued: false, reason: 'no_changed_claims' } },
    onError: error => errors.push(error),
  })
  return { controller, calls, errors, scheduled, advance(value) { current = value; controller.check() } }
}

test('starts a full period after application start and never uses task data to schedule', () => {
  const h = harness()
  h.controller.start()
  h.controller.start()
  assert.equal(h.scheduled.size, 1)
  h.advance(99)
  assert.equal(h.calls.length, 0)
  h.advance(100)
  h.advance(200)
  assert.deepEqual(h.calls.map(call => call.taskId), ['state', 'state'])
  assert.deepEqual(Object.keys(h.calls[0]).sort(), ['signal', 'taskId'])
  h.controller.stop()
})

test('each registered task keeps its own period and delayed checks do not cause catch-up bursts', () => {
  const h = harness([{ taskId: 'a', intervalMs: 100 }, { taskId: 'b', intervalMs: 250 }])
  h.controller.start()
  h.advance(100)
  h.advance(250)
  assert.deepEqual(h.calls.map(call => call.taskId), ['a', 'a', 'b'])
  h.advance(5000)
  assert.deepEqual(h.calls.slice(-2).map(call => call.taskId), ['a', 'b'])
  h.advance(5000)
  assert.equal(h.calls.length, 5)
  h.controller.stop()
})

test('stop cancels signals and timers, and a new run starts a fresh period', () => {
  const h = harness()
  h.controller.start()
  h.advance(100)
  h.controller.stop()
  assert.equal(h.calls[0].signal.aborted, true)
  assert.equal(h.scheduled.size, 0)
  h.advance(1000)
  assert.equal(h.calls.length, 1)
  h.controller.start()
  h.advance(1099)
  assert.equal(h.calls.length, 1)
  h.advance(1100)
  assert.equal(h.calls.length, 2)
  h.controller.stop()
})

test('a failed signal does not stop the clock', async () => {
  let current = 0
  const calls = []
  const errors = []
  const h = createPeriodicTaskController({ tasks: [{ taskId: 'a', intervalMs: 100 }],
    now: () => current, setTimer: () => 1, clearTimer() {},
    onTrigger() { calls.push(current); throw new Error('offline') },
    onError: error => errors.push(error),
  })
  h.start()
  current = 100; h.check()
  current = 200; h.check()
  assert.equal(calls.length, 2)
  assert.equal(errors.length, 2)
  h.stop()
})

test('PWM timer configuration is independent of claim counts', () => {
  assert.deepEqual(desktopPwmPeriodicTasks({}), [{ taskId: 'user_state_from_claims', intervalMs: 1_800_000 }])
  assert.deepEqual(desktopPwmPeriodicTasks({ QWEN_AUDIO_PWM_USER_STATE_INTERVAL_SECONDS: '0' }), [])
  assert.equal(desktopPwmPeriodicTasks({ QWEN_AUDIO_PWM_USER_STATE_INTERVAL_SECONDS: '60' })[0].intervalMs, 60_000)
  assert.throws(() => desktopPwmPeriodicTasks({ QWEN_AUDIO_PWM_USER_STATE_INTERVAL_SECONDS: 'invalid' }))
})

test('transport sends only a task signal with the current Gateway credential', async () => {
  const calls = []
  await sendPwmPeriodicSignal({ origin: 'http://localhost:3101/', accessToken: 'test-device-token',
    taskId: 'user_state_from_claims',
    fetchImpl: async (...args) => { calls.push(args); return { ok: true, json: async () => ({ queued: true }) } },
  })
  assert.equal(calls[0][0], 'http://localhost:3101/api/memory/periodic-task')
  assert.deepEqual(JSON.parse(calls[0][1].body), { taskId: 'user_state_from_claims' })
  assert.equal(calls[0][1].headers.Authorization, 'Bearer test-device-token')
})


test('asynchronous signal failures are reported, while shutdown aborts stay quiet', async () => {
  let current = 0
  const errors = []
  let fail
  const controller = createPeriodicTaskController({
    tasks: [{ taskId: 'a', intervalMs: 100 }], now: () => current,
    setTimer: () => 1, clearTimer() {},
    onTrigger: () => new Promise((_resolve, reject) => { fail = reject }),
    onError: error => errors.push(error),
  })
  controller.start()
  current = 100; controller.check()
  fail(new Error('offline'))
  await Promise.resolve(); await Promise.resolve()
  assert.equal(errors.length, 1)
  current = 200; controller.check()
  controller.stop()
  fail(new Error('cancelled'))
  await Promise.resolve(); await Promise.resolve()
  assert.equal(errors.length, 1)
})

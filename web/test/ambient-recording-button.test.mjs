import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

let server
let AmbientRecordingButton

before(async () => {
  server = await createServer({
    root: fileURLToPath(new URL('../', import.meta.url)),
    configFile: false,
    esbuild: { jsx: 'automatic' },
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { middlewareMode: true, watch: null, hmr: false },
  })
  AmbientRecordingButton = (
    await server.ssrLoadModule('/src/desktop/AmbientRecordingButton.jsx')
  ).default
  const { setRuntimeLanguage } = await server.ssrLoadModule('/src/i18n.js')
  setRuntimeLanguage('zh-CN')
})

after(async () => { await server?.close() })

function event(details = {}) {
  return { stopPropagation() { this.stopped = true }, ...details }
}

test('a pointer gesture starts once and a later gesture stops once, even with fast IPC', async () => {
  let state = 'inactive'
  const requests = []
  const button = () => AmbientRecordingButton({
    state,
    onToggle: () => {
      requests.push(state === 'recording' ? 'stop' : 'start')
      // A fast IPC state event can rerender the button before pointerup/click.
      state = state === 'recording' ? 'inactive' : 'recording'
    },
  })
  for (const expected of [['start'], ['start', 'stop']]) {
    const down = event({ button: 0, pointerType: 'mouse' })
    button().props.onPointerDown(down)
    await Promise.resolve()
    const click = event({ detail: 1 })
    button().props.onClick(click)
    assert.equal(down.stopped, true, 'must not start dragging the orb')
    assert.equal(click.stopped, true)
    assert.deepEqual(requests, expected)
  }
})

test('keyboard and accessibility clicks can start and stop recording', () => {
  for (const state of ['inactive', 'recording', 'error']) {
    let calls = 0
    const button = AmbientRecordingButton({ state, onToggle: () => { calls += 1 } })
    button.props.onClick(event({ detail: 0 }))
    assert.equal(calls, 1)
    assert.equal(button.props['aria-pressed'], state === 'recording')
  }
})

test('startup and finalization disable activation and describe their busy state', () => {
  for (const state of ['starting', 'finalizing']) {
    const button = AmbientRecordingButton({ state, onToggle: () => assert.fail('busy') })
    assert.equal(button.props.disabled, true)
    assert.equal(button.props['aria-busy'], true)
    assert.match(button.props.title, state === 'starting' ? /启动/ : /处理/)
    button.props.onClick(event())
  }
})

test('pointerdown without click never changes recording state', () => {
  const button = AmbientRecordingButton({
    state: 'inactive', onToggle: () => assert.fail('cancelled gesture activated recording'),
  })
  button.props.onPointerDown(event({ button: 0 }))
  button.props.onPointerDown(event({ button: 2 }))
})

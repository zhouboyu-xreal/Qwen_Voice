import assert from 'node:assert/strict'
import test from 'node:test'
import { DesktopPresence } from '../src/desktop-presence.mjs'

function harness() {
  const events = []
  const callbacks = new Map()
  const window = {
    hidden: false,
    shown: 0,
    focused: 0,
    isDestroyed: () => false,
    isMinimized: () => false,
    show() { this.shown += 1 },
    focus() { this.focused += 1 },
    hide() { this.hidden = true },
    webContents: {
      send: (_channel, event) => events.push(event),
    },
  }
  const globalShortcut = {
    register: (value, callback) => {
      callbacks.set(value, callback)
      return value !== 'unavailable'
    },
    unregister: value => callbacks.delete(value),
    unregisterAll: () => callbacks.clear(),
  }
  const presence = new DesktopPresence({
    getWindow: () => window,
    globalShortcut,
  })
  return { callbacks, events, presence, window }
}

test('starts sleeping and keeps the orb visible through repeated keyword activations', () => {
  const { events, presence, window } = harness()
  assert.equal(presence.state, 'sleeping')
  for (let i = 0; i < 2; i += 1) {
    assert.equal(presence.wake('wake-word'), true)
    assert.deepEqual(events.at(-1), { state: 'waking', reason: 'wake-word' })
    assert.equal(presence.wake('wake-word'), false)
    assert.equal(presence.ready(), true)
    assert.equal(presence.hide(), 'sleeping')
    assert.equal(window.hidden, false)
    assert.deepEqual(events.at(-1), { state: 'sleeping', reason: 'inactivity' })
  }
})

test('presentation requests never activate the sleeping microphone', () => {
  const { callbacks, presence, window } = harness()
  presence.registerShortcut('test-shortcut')
  callbacks.get('test-shortcut')()
  for (const reason of ['tray', 'panel', 'settings', 'activate', 'orb', 'second-instance']) {
    assert.equal(presence.wake(reason), true)
    assert.equal(presence.state, 'sleeping')
    assert.equal(presence.ready(), false)
  }
  assert.equal(window.shown, 7)
  assert.equal(window.hidden, false)
})

test('sleep cancels a pending wake and resets the local detector once', () => {
  const { presence } = harness()
  let resets = 0
  presence.onSleep = () => { resets += 1 }
  presence.wake('wake-word')
  assert.equal(presence.sleep('gateway-disconnected'), 'sleeping')
  assert.equal(presence.ready(), false)
  assert.equal(presence.sleep(), 'sleeping')
  assert.equal(resets, 1)
})

test('keeps the previous shortcut when a replacement is unavailable', () => {
  const { callbacks, presence } = harness()
  assert.equal(presence.registerShortcut('old'), true)
  assert.equal(presence.registerShortcut('unavailable'), false)
  assert.equal(presence.shortcut, 'old')
  assert.equal(callbacks.has('old'), true)
})

test('temporarily releases and restores the configured shortcut', () => {
  const { callbacks, presence } = harness()
  presence.registerShortcut('wake')
  presence.pauseShortcut()
  assert.equal(callbacks.has('wake'), false)
  assert.equal(presence.shortcutPaused, true)
  assert.equal(presence.resumeShortcut(), true)
  assert.equal(callbacks.has('wake'), true)
})

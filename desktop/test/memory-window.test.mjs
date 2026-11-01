import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import test from 'node:test'
import { createMemoryWindowController } from '../src/memory-window.mjs'

class FakeWindow extends EventEmitter {
  constructor(options) {
    super()
    this.options = options
    this.webContents = new EventEmitter()
    this.webContents.setWindowOpenHandler = callback => { this.openHandler = callback }
    this.webContents.setAudioMuted = muted => { this.muted = muted }
    this.shown = 0
    this.focused = 0
    this.loads = []
  }
  isDestroyed() { return this.destroyed === true }
  isMinimized() { return this.minimized === true }
  restore() { this.minimized = false }
  show() { this.shown += 1 }
  focus() { this.focused += 1 }
  setTitle(title) { this.title = title }
  async loadURL(url) { this.loads.push(url) }
  destroy() { this.destroyed = true; this.emit('closed') }
}

test('reuses an independent silent window without a voice preload', async () => {
  let url = 'http://127.0.0.1:1234/relay/?view=memory'
  const control = createMemoryWindowController({
    BrowserWindow: FakeWindow, getUrl: () => url, getTitle: () => '我的记忆',
    getWorkArea: () => ({ width: 700, height: 600 }),
  })
  assert.equal(await control.reload(), false)
  assert.equal(await control.open(), true)
  const window = control.window()
  assert.equal(window.options.width, 700)
  assert.equal(window.options.height, 600)
  assert.equal(window.options.webPreferences.preload, undefined)
  assert.equal(window.options.webPreferences.nodeIntegration, false)
  assert.equal(window.muted, true)
  window.minimized = true
  await control.open()
  assert.equal(control.window(), window)
  assert.equal(window.minimized, false)
  assert.equal(window.loads.length, 1)
  url = 'http://127.0.0.1:1234/relay/?view=memory&lang=en'
  await control.reload()
  assert.equal(window.loads.at(-1), url)
  assert.deepEqual(window.openHandler(), { action: 'deny' })
  for (const target of ['https://example.test', 'http://127.0.0.1:1234/relay/?desktop=orb']) {
    let prevented = false
    window.webContents.emit('will-navigate', { preventDefault: () => { prevented = true } }, target)
    assert.equal(prevented, true)
  }
  control.close()
  assert.equal(control.window(), null)
  await control.open()
  assert.notEqual(control.window(), window)
  control.close()
})

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  applyDesktopClientSettings,
  initialDesktopClientSettings,
} from '../src/desktop/desktop-client-settings.js'

test('desktop client settings initialize from the desktop URL', () => {
  assert.deepEqual(initialDesktopClientSettings(
    '?orbSkin=firefly&autoHideSeconds=300&wakeWordEnabled=true&lang=en',
  ), {
    orbSkinId: 'firefly',
    autoHideSeconds: 300,
    wakeWordEnabled: true,
    language: 'en',
    microphoneEchoCancellation: 'all',
  })
})

test('desktop client settings hot-apply without replacing unrelated state', () => {
  const current = {
    orbSkinId: 'fluid',
    autoHideSeconds: 60,
    wakeWordEnabled: true,
    language: 'zh-CN',
  }

  assert.deepEqual(applyDesktopClientSettings(current, {
    orbSkin: 'firefly',
    autoHideSeconds: 120,
    language: 'en',
    microphoneEchoCancellation: 'default',
  }), {
    orbSkinId: 'firefly',
    autoHideSeconds: 120,
    wakeWordEnabled: true,
    language: 'en',
    microphoneEchoCancellation: 'all',
  })
})


test('legacy settings cannot turn off keyword gating', () => {
  assert.equal(initialDesktopClientSettings('?wakeWordEnabled=false').wakeWordEnabled, true)
  assert.equal(applyDesktopClientSettings(initialDesktopClientSettings(), {
    wakeWordEnabled: false,
  }).wakeWordEnabled, true)
})

test('desktop always requests all despite legacy URL and hot settings overrides', () => {
  const current = initialDesktopClientSettings('?microphoneEchoCancellation=default')
  assert.equal(current.microphoneEchoCancellation, 'all')
  assert.equal(applyDesktopClientSettings(current, { microphoneEchoCancellation: 'default' }).microphoneEchoCancellation, 'all')
})

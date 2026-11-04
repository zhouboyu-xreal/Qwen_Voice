import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { createMicrophoneDiagnosticStore, sanitizeMicrophoneDiagnostics } from '../src/microphone-diagnostics.mjs'
import { createSettingsStore } from '../src/settings-store.mjs'
import { parseSettings, updateSettingsContent } from '../src/settings-config.mjs'
const payload = { captureId: 'capture-1', phase: 'opened', requestedMode: 'all', appliedMode: 'all', allSupport: 'confirmed',
  device: { label: 'MacBook microphone', deviceId: 'secret-device' },
  settings: { echoCancellation: 'all', sampleRate: 48000, noiseSuppression: true, token: 'secret' },
  capabilities: { echoCancellation: [true, false, 'all'], sampleRate: { min: 48000, max: 48000 } },
  supportedConstraints: { echoCancellation: true },
}
function temporary(t) {
  const root = mkdtempSync(join(tmpdir(), 'microphone-diagnostics-'))
  t.after(() => rmSync(root, { recursive: true, force: true }))
  return root
}
test('only known capture fields enter desktop logs and diagnostic artifacts', () => {
  const safe = sanitizeMicrophoneDiagnostics({ ...payload, audio: 'private-pcm', token: 'secret-token' })
  assert.equal(safe.settings.echoCancellation, 'all')
  assert.doesNotMatch(JSON.stringify(safe), /private-pcm|secret|deviceId|token/)
  const unconfirmed = sanitizeMicrophoneDiagnostics({ ...payload, settings: { echoCancellation: true } })
  assert.equal(unconfirmed.allSupport, 'unconfirmed')
  assert.equal(unconfirmed.appliedMode, 'default')
})
test('capture settings and device changes stay linked to the recording and are saved privately', t => {
  const root = temporary(t)
  const events = []
  const store = createMicrophoneDiagnosticStore({ logger: { info: (...args) => events.push(args) }, runtime: { chrome: 'test-runtime' } })
  store.record(payload)
  store.begin({ recordingId: 'recording-1', directory: root, requestedMode: 'all' })
  store.record({ ...payload, captureId: 'capture-2', device: { label: 'USB microphone' } })
  store.end('stale-recording')
  store.end('recording-1')
  const path = join(root, 'recording-1.microphone.json')
  const report = JSON.parse(readFileSync(path))
  assert.equal(report.state, 'stopped')
  assert.equal(report.captures.length, 2)
  assert.equal(report.captures[1].device.label, 'USB microphone')
  assert.equal(events[1][1].recordingId, 'recording-1')
  assert.equal(statSync(path).mode & 0o777, 0o600)
})
test('legacy AEC preferences are ignored and no longer saved by settings', t => {
  const root = temporary(t)
  const store = createSettingsStore({ configDir: join(root, 'gateway'), clientDir: join(root, 'client'), env: {} })
  store.save({ dashscopeApiKey: 'fixture-key', ambientRecordingSource: 'microphone-system', microphoneEchoCancellation: 'default' })
  assert.equal(Object.hasOwn(store.load(), 'microphoneEchoCancellation'), false)
  assert.doesNotMatch(readFileSync(store.clientSettingsPath, 'utf8'), /MICROPHONE_ECHO/)
  assert.doesNotMatch(readFileSync(store.path, 'utf8'), /MICROPHONE_ECHO/)
  writeFileSync(store.clientSettingsPath, 'QWEN_AUDIO_MICROPHONE_ECHO_CANCELLATION=default\n')
  assert.equal(Object.hasOwn(store.load(), 'microphoneEchoCancellation'), false)
  assert.equal(Object.hasOwn(parseSettings('', { QWEN_AUDIO_MICROPHONE_ECHO_CANCELLATION: 'default' }), 'microphoneEchoCancellation'), false)
  assert.doesNotMatch(updateSettingsContent('', { microphoneEchoCancellation: 'default' }), /MICROPHONE_ECHO/)
})

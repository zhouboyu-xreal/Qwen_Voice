import assert from 'node:assert/strict'
import test from 'node:test'
import { acquireMicrophoneWithDiagnostics } from '../src/realtime/microphone-aec.js'
function media(mode = true) {
  const track = { label: 'Built-in microphone', stopped: 0, stop() { this.stopped++ },
    getSettings: () => ({ echoCancellation: mode, sampleRate: 48000, noiseSuppression: true,
      deviceId: 'private-device', groupId: 'private-group' }),
    getCapabilities: () => ({ echoCancellation: [false, true, 'all'], sampleRate: { min: 8000, max: 48000 }, deviceId: 'private-device' }),
  }
  return { track, getAudioTracks: () => [track], getTracks: () => [track] }
}
function devices(acquire) { return { getUserMedia: acquire, getSupportedConstraints: () => ({ echoCancellation: true }) } }
test('control retains browser default and reports advertised capability without claiming all was used', async () => {
  const calls = []
  const result = await acquireMicrophoneWithDiagnostics({ mode: 'default', mediaDevices: devices(async constraints => { calls.push(constraints); return media() }) })
  assert.equal(calls[0].audio.echoCancellation, true)
  assert.equal(result.diagnostics.appliedMode, 'default')
  assert.equal(result.diagnostics.allSupport, 'advertised')
  assert.equal(result.diagnostics.settings.echoCancellation, true)
  assert.doesNotMatch(JSON.stringify(result.diagnostics), /private-device|private-group|deviceId|groupId/)
})
test('all is confirmed only after strict negotiation and settings report all', async () => {
  const calls = []
  const result = await acquireMicrophoneWithDiagnostics({ mode: 'all', deviceId: 'chosen', mediaDevices: devices(async c => { calls.push(c); return media('all') }) })
  assert.deepEqual(calls[0].audio.echoCancellation, { exact: 'all' })
  assert.deepEqual(calls[0].audio.deviceId, { exact: 'chosen' })
  assert.equal(result.diagnostics.appliedMode, 'all')
  assert.equal(result.diagnostics.allSupport, 'confirmed')
})
test('unsupported all falls back once and keeps the rejected mode visible in diagnostics', async () => {
  const calls = []
  const result = await acquireMicrophoneWithDiagnostics({ mode: 'all', mediaDevices: devices(async c => {
    calls.push(c)
    if (calls.length === 1) throw Object.assign(new Error('not supported'), { name: 'OverconstrainedError', constraint: 'echoCancellation' })
    return media()
  }) })
  assert.equal(calls.length, 2)
  assert.equal(calls[1].audio.echoCancellation, true)
  assert.equal(result.diagnostics.requestedMode, 'all')
  assert.equal(result.diagnostics.appliedMode, 'default')
  assert.equal(result.diagnostics.allSupport, 'rejected')
})
test('a browser ignoring the all constraint releases that stream before fallback', async () => {
  const first = media(true)
  const second = media(true)
  let calls = 0
  const result = await acquireMicrophoneWithDiagnostics({ mode: 'all', mediaDevices: devices(async () => ++calls === 1 ? first : second) })
  assert.equal(first.track.stopped, 1)
  assert.equal(second.track.stopped, 0)
  assert.equal(result.diagnostics.allSupport, 'unconfirmed')
  assert.equal(result.diagnostics.fallbackReason, 'all-not-reported')
})
test('permission and unrelated device constraints fail without disguising them as unsupported all', async () => {
  for (const error of [Object.assign(new Error('denied'), { name: 'NotAllowedError' }),
    Object.assign(new Error('device gone'), { name: 'OverconstrainedError', constraint: 'deviceId' })]) {
    let calls = 0
    const reports = []
    await assert.rejects(acquireMicrophoneWithDiagnostics({ mode: 'all', mediaDevices: devices(async () => { calls++; throw error }), onDiagnostics: d => reports.push(d) }), e => e === error)
    assert.equal(calls, 1)
    assert.equal(reports[0].phase, 'failed')
  }
})
test('missing capability APIs and a failed logging callback do not break default capture', async () => {
  const value = media()
  delete value.track.getCapabilities
  const result = await acquireMicrophoneWithDiagnostics({ mediaDevices: { getUserMedia: async () => value }, onDiagnostics: () => { throw new Error('logger unavailable') } })
  assert.equal(result.media, value)
  assert.deepEqual(result.diagnostics.capabilities, {})
})

test('nominal all support with no PCM releases the stalled track and verifies the default fallback', async () => {
  const first = media('all')
  const fallback = media(true)
  const verified = []
  let calls = 0
  const result = await acquireMicrophoneWithDiagnostics({ mode: 'all', mediaDevices: devices(async () => ++calls === 1 ? first : fallback),
    verifyFrames: async value => { verified.push(value); return value === fallback } })
  assert.deepEqual(verified, [first, fallback])
  assert.equal(first.track.stopped, 1)
  assert.equal(fallback.track.stopped, 0)
  assert.equal(result.diagnostics.appliedMode, 'default')
  assert.equal(result.diagnostics.allSupport, 'no-pcm')
  assert.equal(result.diagnostics.allPcmFlow, 'timeout')
  assert.equal(result.diagnostics.allNegotiatedEchoCancellation, 'all')
  assert.equal(result.diagnostics.fallbackReason, 'all-no-pcm')
})
test('a stalled fallback reports an input failure instead of microphone readiness', async () => {
  const first = media('all')
  const fallback = media(true)
  let calls = 0
  await assert.rejects(acquireMicrophoneWithDiagnostics({ mode: 'all', mediaDevices: devices(async () => ++calls === 1 ? first : fallback),
    verifyFrames: async () => false }), { name: 'NotReadableError' })
  assert.equal(first.track.stopped, 1)
  assert.equal(fallback.track.stopped, 1)
})
test('the PCM probe treats silent audio frames as valid and cleans up its graph on timeout', async () => {
  const { waitForMicrophoneFrames } = await import('../src/realtime/microphone-aec.js')
  const nodes = []
  let processor
  const node = () => { const value = { gain: {}, disconnected: false, connect() {}, disconnect() { this.disconnected = true } }; nodes.push(value); return value }
  const context = { createMediaStreamSource: node, createScriptProcessor: () => (processor = node()), createGain: node, destination: {} }
  const pending = waitForMicrophoneFrames({ media: {}, context, timeoutMs: 100 })
  processor.onaudioprocess({ inputBuffer: { length: 2048 } })
  assert.equal(await pending, true)
  assert.ok(nodes.every(node => node.disconnected))
  assert.equal(await waitForMicrophoneFrames({ media: {}, context, timeoutMs: 5 }), false)
  assert.ok(nodes.every(node => node.disconnected))
})

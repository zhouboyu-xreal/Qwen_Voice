import assert from 'node:assert/strict'
import test from 'node:test'
import { configureSystemAudioCapture, systemAudioCaptureSupported } from '../src/system-audio-capture.mjs'
import { parseSettings, normalizeSettings, updateSettingsContent, clientSettingsPatch } from '../src/settings-config.mjs'
import { AmbientAudioRecordingRuntime } from '../../shared/voice/ambient-audio-recording-runtime.mjs'

test('system capture capability is explicit and does not pretend Linux has a loopback track', () => {
  for (const [platform, version, supported] of [
    ['darwin', '14.1', false], ['darwin', '14.2', true], ['darwin', '15.0', true],
    ['darwin', '', false], ['win32', '', true], ['linux', '', false],
  ]) assert.equal(systemAudioCaptureSupported(platform, version), supported)
})

test('capture requests are restricted to the authorized frame and rechecked after enumeration', async () => {
  let handler
  let options
  let permitted = true
  let enumerations = 0
  const frame = {}
  configureSystemAudioCapture({
    platform: 'win32',
    session: { setDisplayMediaRequestHandler: (fn, opts) => { handler = fn; options = opts } },
    desktopCapturer: { getSources: async () => { enumerations++; return [{ id: 'screen-fixture' }] } },
    allowed: candidate => candidate === frame && permitted,
  })
  assert.deepEqual(options, { useSystemPicker: false })
  let grant
  await handler({ frame: {} }, result => { grant = result })
  assert.deepEqual(grant, {})
  assert.equal(enumerations, 0)
  await handler({ frame }, result => { grant = result })
  assert.deepEqual(grant, { video: { id: 'screen-fixture' }, audio: 'loopback' })
  const pending = handler({ frame }, result => { grant = result })
  permitted = false
  await pending
  assert.deepEqual(grant, {})
})

test('source preference is a private desktop setting and survives round-trip with defaults', () => {
  assert.equal(parseSettings('').ambientRecordingSource, 'microphone-system')
  const content = updateSettingsContent('UNRELATED=kept\n', { ambientRecordingSource: 'microphone' })
  assert.equal(parseSettings(content).ambientRecordingSource, 'microphone')
  assert.match(content, /UNRELATED=kept/)
  assert.deepEqual(clientSettingsPatch({ ambientRecordingSource: 'microphone', agentProtocol: 'none' }), { ambientRecordingSource: 'microphone' })
  assert.throws(() => normalizeSettings({ ambientRecordingSource: 'system' }), /环境录音来源/)
})

test('sidecar transport preserves track identity and capture timestamps', async () => {
  const runtime = new AmbientAudioRecordingRuntime()
  const messages = []
  runtime.request = async (method, params) => { messages.push({ method, params }); return {} }
  await runtime.startRecording({ recordingId: 'session', startedAt: '2026-11-03T01:00:00Z', sources: ['microphone', 'system'] })
  await runtime.appendPcm16(Buffer.from([1, 0]), 48000, 'session', { source: 'system', startedAt: '2026-11-03T01:00:01Z' })
  assert.deepEqual(messages[0].params.sources, ['microphone', 'system'])
  assert.deepEqual(messages[1], { method: 'audio.append', params: {
    recordingId: 'session', audio: 'AQA=', sampleRate: 48000, source: 'system', startedAt: '2026-11-03T01:00:01Z',
  } })
})


test('macOS never enumerates screens or invokes its system screen picker', async () => {
  let handler
  configureSystemAudioCapture({
    platform: 'darwin', session: { setDisplayMediaRequestHandler: (fn, options) => {
      handler = fn; assert.equal(options.useSystemPicker, false)
    } },
    desktopCapturer: { getSources: () => assert.fail('macOS must not enumerate screens') },
    allowed: () => true,
  })
  await handler({ frame: {} }, result => assert.deepEqual(result, {}))
})

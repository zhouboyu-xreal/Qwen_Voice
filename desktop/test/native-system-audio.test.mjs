import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, rm, writeFile, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { NativeSystemAudioCapture, nativeSystemAudioExecutable } from '../src/native-system-audio.mjs'
import { parse } from 'yaml'

async function fixture(t, body, options = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'qwen-native-audio-test-'))
  const path = join(directory, 'helper.mjs')
  await writeFile(path, `import { createInterface } from 'node:readline';
const emit = value => process.stdout.write(JSON.stringify(value) + '\\n');
const input = createInterface({ input: process.stdin });
${body}`)
  const capture = new NativeSystemAudioCapture({
    executable: process.execPath,
    spawnImpl: (executable, _args, opts) => spawn(executable, [path], opts),
    startTimeoutMs: 2000, stopTimeoutMs: 2000,
    ...options,
  })
  t.after(async () => { await capture.stop().catch(() => {}); await rm(directory, { recursive: true, force: true }) })
  return capture
}

const healthy = `
emit({ type: 'ready', sampleRate: 48000 });
input.on('line', line => {
  if (line === 'begin') emit({ type: 'audio', audio: 'AQACAA==', sampleRate: 48000, startedAt: '2026-11-03T08:00:00.000Z' });
  if (line === 'stop') {
    emit({ type: 'audio', audio: 'AwA=', sampleRate: 48000, startedAt: '2026-11-03T08:00:00.001Z' });
    emit({ type: 'stopped' });
    input.close(); process.stdin.destroy();
  }
});`

test('native transport waits for final PCM persistence before stop resolves', async t => {
  const packets = []
  let sawFirst
  let sawTail
  let persist
  const first = new Promise(resolve => { sawFirst = resolve })
  const tail = new Promise(resolve => { sawTail = resolve })
  const persistence = new Promise(resolve => { persist = resolve })
  const capture = await fixture(t, healthy, {
    onAudio: async (audio, sampleRate, startedAt) => {
      packets.push({ audio, sampleRate, startedAt })
      if (packets.length === 1) sawFirst()
      else sawTail()
      await persistence
    },
  })
  const ready = await capture.start()
  assert.equal(ready.backend, 'native')
  capture.begin()
  await first
  let stopped = false
  const ending = capture.stop().then(() => { stopped = true })
  await tail
  assert.equal(stopped, false)
  persist()
  await ending
  assert.deepEqual(packets.map(packet => [...packet.audio]), [[1, 0, 2, 0], [3, 0]])
  assert.ok(packets.every(packet => packet.sampleRate === 48000 && packet.startedAt.endsWith('Z')))
})

test('native startup errors are structured, do not expose diagnostics and never deliver PCM', async t => {
  const capture = await fixture(t, `
process.stderr.write('fixture-private-diagnostic');
emit({ type: 'error', code: 'create_tap', status: 123 });
input.on('line', () => { input.close(); process.stdin.destroy(); });`, {
    onAudio: () => assert.fail('failed startup must not deliver audio'),
  })
  await assert.rejects(capture.start(), error => {
    assert.match(error.message, /系统音频录制/)
    assert.doesNotMatch(error.message, /fixture-private/)
    return true
  })
  assert.equal(capture.ready, false)
})

test('an unexpected helper exit reports failure and closes the recorder session', async t => {
  let failed
  const failure = new Promise(resolve => { failed = resolve })
  const capture = await fixture(t, `
emit({ type: 'ready', sampleRate: 48000 });
input.on('line', () => process.exit(1));`, { onAudio() {}, onError: failed })
  await capture.start()
  capture.begin()
  assert.match((await failure).message, /系统音频采集失败/)
  await assert.rejects(capture.stop())
})

test('native output device changes request a fresh recording rather than corrupting a WAV format', async t => {
  let failed
  const failure = new Promise(resolve => { failed = resolve })
  const capture = await fixture(t, `
emit({ type: 'ready', sampleRate: 48000 });
input.on('line', line => {
  if (line === 'begin') emit({ type: 'error', code: 'output_device_changed', status: 0 });
  if (line === 'stop') { input.close(); process.stdin.destroy(); }
});`, { onAudio() {}, onError: failed })
  await capture.start()
  capture.begin()
  assert.match((await failure).message, /音频输出设备已切换/)
})

test('binary paths and release configuration include an external signed helper', async () => {
  assert.equal(nativeSystemAudioExecutable({ packaged: true, resourcesPath: '/bundle/Resources', sourceRoot: '/repo' }), '/bundle/Resources/native/qwen-system-audio')
  assert.equal(nativeSystemAudioExecutable({ packaged: false, sourceRoot: '/repo', arch: 'x64' }), '/repo/desktop/native/bin/x64/qwen-system-audio')
  const config = parse(await readFile(new URL('../electron-builder.yml', import.meta.url), 'utf8'))
  assert.equal(config.beforePack, 'desktop/native/before-pack.cjs')
  assert.deepEqual(config.mac.extraResources, [{ from: 'desktop/native/bin/${arch}/qwen-system-audio', to: 'native/qwen-system-audio' }])
  assert.ok(config.mac.binaries.includes('Contents/Resources/native/qwen-system-audio'))
  assert.ok(config.mac.extendInfo.NSAudioCaptureUsageDescription)
})

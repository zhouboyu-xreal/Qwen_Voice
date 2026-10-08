import assert from 'node:assert/strict'
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  VoiceReferenceRecordingRuntime,
} from '../../shared/voice/voice-reference-recording-runtime.mjs'

test('VoiceReferenceRecordingRuntime saves mono PCM16 as a valid WAV', () => {
  const directory = mkdtempSync(join(tmpdir(), 'qwaudio-voice-reference-'))
  try {
    const runtime = new VoiceReferenceRecordingRuntime({
      recordingsDirectory: directory,
    })
    const started = runtime.startRecording({
      recordingId: 'abcdef12-0000-0000-0000-000000000000',
      startedAt: '2026-10-08T10:30:00.000Z',
    })
    runtime.appendPcm16(Buffer.from([0, 0, 1, 0, 255, 127, 0, 128]), 16_000)
    const saved = runtime.stopRecording()

    assert.equal(started.state, 'recording')
    assert.equal(saved.state, 'saved')
    assert.equal(saved.sampleRate, 16_000)
    assert.equal(saved.durationSeconds, 4 / 16_000)
    assert.equal(existsSync(saved.path), true)
    const data = readFileSync(saved.path)
    assert.equal(data.subarray(0, 4).toString('ascii'), 'RIFF')
    assert.equal(data.subarray(8, 12).toString('ascii'), 'WAVE')
    assert.equal(data.readUInt16LE(22), 1)
    assert.equal(data.readUInt16LE(34), 16)
    assert.equal(data.readUInt32LE(24), 16_000)
    assert.equal(data.readUInt32LE(40), 8)
    assert.equal(data.length, 52)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('VoiceReferenceRecordingRuntime discards an empty capture', () => {
  const directory = mkdtempSync(join(tmpdir(), 'qwaudio-voice-reference-'))
  try {
    const runtime = new VoiceReferenceRecordingRuntime({
      recordingsDirectory: directory,
    })
    runtime.startRecording({ recordingId: 'empty-capture' })
    assert.deepEqual(runtime.stopRecording(), {
      state: 'inactive',
      discarded: true,
    })
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('VoiceReferenceRecordingRuntime concatenates only VAD speech segments', () => {
  const directory = mkdtempSync(join(tmpdir(), 'qwaudio-voice-reference-'))
  const calls = []
  const vad = {
    acceptPcm16(chunk, sampleRate) {
      calls.push({ chunk: Buffer.from(chunk), sampleRate })
      return [{ startSample: 0, endSample: 2 }]
    },
    flush() {
      return [{ startSample: 2, endSample: 4 }]
    },
    close() {
      calls.push({ closed: true })
    },
  }
  try {
    const runtime = new VoiceReferenceRecordingRuntime({
      recordingsDirectory: directory,
      vad,
    })
    runtime.startRecording({ recordingId: 'vad-capture' })
    runtime.appendPcm16(Buffer.from([1, 0, 2, 0, 3, 0, 4, 0]), 16_000)
    const saved = runtime.stopRecording()

    assert.equal(saved.state, 'saved')
    assert.equal(saved.speechSegmentCount, 2)
    assert.equal(saved.sourceDurationSeconds, 4 / 16_000)
    assert.deepEqual(calls[0], {
      chunk: Buffer.from([1, 0, 2, 0, 3, 0, 4, 0]),
      sampleRate: 16_000,
    })
    assert.equal(calls.at(-1).closed, true)
    const output = readFileSync(saved.path)
    assert.deepEqual(output.subarray(44), Buffer.from([1, 0, 2, 0, 3, 0, 4, 0]))
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('VoiceReferenceRecordingRuntime writes detected speech at the source sample rate', () => {
  const directory = mkdtempSync(join(tmpdir(), 'qwaudio-voice-reference-'))
  const vad = {
    acceptPcm16() { return [{ startSample: 0, endSample: 160 }] },
    flush() { return [] },
  }
  try {
    const runtime = new VoiceReferenceRecordingRuntime({
      recordingsDirectory: directory,
      vad,
    })
    runtime.startRecording({ recordingId: 'resampled-capture' })
    runtime.appendPcm16(Buffer.alloc(48_000 * 2), 48_000)
    const saved = runtime.stopRecording()

    assert.equal(saved.state, 'saved')
    assert.equal(saved.sampleRate, 48_000)
    assert.equal(saved.sourceDurationSeconds, 1)
    const output = readFileSync(saved.path)
    assert.equal(output.readUInt32LE(24), 48_000)
    assert.equal(output.readUInt32LE(40), 960)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

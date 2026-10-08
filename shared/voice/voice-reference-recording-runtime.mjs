import {
  closeSync,
  mkdirSync,
  openSync,
  renameSync,
  rmSync,
  writeSync,
} from 'node:fs'
import { basename, join, resolve } from 'node:path'

const WAV_HEADER_BYTES = 44
const PCM_BITS_PER_SAMPLE = 16
const PCM_CHANNELS = 1
const DEFAULT_SAMPLE_RATE = 16_000
const VAD_SAMPLE_RATE = 16_000
const MAX_BUFFERED_SOURCE_SECONDS = 35

function clean(value, limit = 128) {
  return [...String(value || '').replaceAll('\0', '').trim()]
    .slice(0, limit)
    .join('')
}

function timestampForFilename(value) {
  const parsed = new Date(value)
  const date = Number.isNaN(parsed.getTime()) ? new Date() : parsed
  const pad = number => String(number).padStart(2, '0')
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}`
    + `_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
}

function normalizeSampleRate(value) {
  const sampleRate = Number(value)
  if (!Number.isInteger(sampleRate) || sampleRate < 8_000 || sampleRate > 192_000) {
    throw new RangeError('voice reference recording sample rate is invalid')
  }
  return sampleRate
}

function wavHeader({ dataBytes = 0, sampleRate = DEFAULT_SAMPLE_RATE } = {}) {
  const header = Buffer.alloc(WAV_HEADER_BYTES)
  const byteRate = sampleRate * PCM_CHANNELS * (PCM_BITS_PER_SAMPLE / 8)
  const blockAlign = PCM_CHANNELS * (PCM_BITS_PER_SAMPLE / 8)
  header.write('RIFF', 0, 'ascii')
  header.writeUInt32LE(36 + dataBytes, 4)
  header.write('WAVE', 8, 'ascii')
  header.write('fmt ', 12, 'ascii')
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(PCM_CHANNELS, 22)
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(byteRate, 28)
  header.writeUInt16LE(blockAlign, 32)
  header.writeUInt16LE(PCM_BITS_PER_SAMPLE, 34)
  header.write('data', 36, 'ascii')
  header.writeUInt32LE(dataBytes, 40)
  return header
}

/**
 * Retains just enough original PCM to turn delayed VAD results back into
 * native-rate speech clips. VAD finalizes a segment only after it observes
 * trailing silence, so writing input eagerly would retain unwanted silence.
 */
class Pcm16Timeline {
  constructor({ sampleRate, maxBufferedSeconds = MAX_BUFFERED_SOURCE_SECONDS }) {
    this.sampleRate = sampleRate
    this.maxBufferedSamples = Math.ceil(sampleRate * maxBufferedSeconds)
    this.startSample = 0
    this.endSample = 0
    this.chunks = []
  }

  append(chunk) {
    const sampleCount = chunk.length / 2
    this.chunks.push({
      chunk,
      startSample: this.endSample,
      endSample: this.endSample + sampleCount,
    })
    this.endSample += sampleCount
  }

  slice(startSample, endSample) {
    const start = Math.max(this.startSample, Math.floor(startSample))
    const end = Math.min(this.endSample, Math.ceil(endSample))
    if (end <= start) return Buffer.alloc(0)
    const output = []
    for (const entry of this.chunks) {
      const overlapStart = Math.max(start, entry.startSample)
      const overlapEnd = Math.min(end, entry.endSample)
      if (overlapEnd <= overlapStart) continue
      const offset = (overlapStart - entry.startSample) * 2
      const byteLength = (overlapEnd - overlapStart) * 2
      output.push(entry.chunk.subarray(offset, offset + byteLength))
    }
    return output.length === 1 ? output[0] : Buffer.concat(output)
  }

  discardBefore(sample) {
    const boundary = Math.min(this.endSample, Math.max(this.startSample, Math.floor(sample)))
    if (boundary <= this.startSample) return
    const retained = []
    for (const entry of this.chunks) {
      if (entry.endSample <= boundary) continue
      if (entry.startSample < boundary) {
        retained.push({
          chunk: entry.chunk.subarray((boundary - entry.startSample) * 2),
          startSample: boundary,
          endSample: entry.endSample,
        })
      } else {
        retained.push(entry)
      }
    }
    this.chunks = retained
    this.startSample = boundary
  }

  retainRecent() {
    this.discardBefore(this.endSample - this.maxBufferedSamples)
  }
}

/**
 * Writes a microphone PCM16 stream into one durable, local WAV file.
 *
 * This runtime deliberately has no ASR, Gateway, or Agent Memory dependency:
 * it records a user-provided voice reference only. A partial suffix prevents a
 * power loss or a renderer crash from being mistaken for a complete WAV file.
 */
export class VoiceReferenceRecordingRuntime {
  constructor({ recordingsDirectory, vad = null } = {}) {
    this.recordingsDirectory = String(recordingsDirectory || '')
    this.vad = vad
    this.active = null
  }

  startRecording({ recordingId, startedAt } = {}) {
    if (this.active) {
      return {
        state: 'recording',
        recordingId: this.active.recordingId,
        path: this.active.finalPath,
      }
    }
    if (!this.recordingsDirectory) {
      throw new Error('voice reference recording directory is required')
    }
    const normalizedId = clean(recordingId)
    if (!normalizedId) throw new Error('voice reference recording id is required')

    const directory = resolve(this.recordingsDirectory)
    mkdirSync(directory, { recursive: true, mode: 0o700 })
    const name = `${timestampForFilename(startedAt)}_voice_reference_${normalizedId.slice(0, 6)}.wav`
    const finalPath = join(directory, name)
    const partialPath = `${finalPath}.partial`
    const fd = openSync(partialPath, 'w', 0o600)
    try {
      writeSync(fd, wavHeader())
    } catch (error) {
      closeSync(fd)
      rmSync(partialPath, { force: true })
      throw error
    }
    this.active = {
      recordingId: normalizedId,
      fd,
      partialPath,
      finalPath,
      sampleRate: 0,
      sourceSampleRate: 0,
      dataBytes: 0,
      sourceBytes: 0,
      sourceSamples: 0,
      lastEmittedSourceSample: 0,
      sourceTimeline: null,
      speechSegmentCount: 0,
      startedAt: String(startedAt || new Date().toISOString()),
    }
    return { state: 'recording', recordingId: normalizedId, path: finalPath }
  }

  appendPcm16(chunk, sampleRate) {
    const active = this.active
    if (!active || !Buffer.isBuffer(chunk) || !chunk.length) return null
    if (chunk.length % 2 !== 0) {
      throw new RangeError('voice reference recording PCM16 chunk has an odd byte length')
    }
    const normalizedRate = normalizeSampleRate(sampleRate)
    if (active.sourceSampleRate && active.sourceSampleRate !== normalizedRate) {
      throw new RangeError('voice reference recording sample rate changed mid-stream')
    }
    active.sourceSampleRate = normalizedRate
    active.sampleRate = normalizedRate
    active.sourceBytes += chunk.length
    active.sourceSamples += chunk.length / 2
    if (!this.vad) return this.#appendSpeechChunks(active, [chunk])

    active.sourceTimeline ??= new Pcm16Timeline({ sampleRate: normalizedRate })
    active.sourceTimeline.append(chunk)
    const detectedSegments = this.vad.acceptPcm16(chunk, normalizedRate)
    const result = this.#appendDetectedSpeechSegments(active, detectedSegments)
    active.sourceTimeline.retainRecent()
    return result
  }

  stopRecording() {
    const active = this.active
    if (!active) return { state: 'inactive' }
    this.active = null
    try {
      if (this.vad) {
        this.#appendDetectedSpeechSegments(active, this.vad.flush())
      }
      if (!active.dataBytes) {
        closeSync(active.fd)
        rmSync(active.partialPath, { force: true })
        return { state: 'inactive', discarded: true }
      }
      const sampleRate = active.sampleRate || DEFAULT_SAMPLE_RATE
      writeSync(active.fd, wavHeader({
        dataBytes: active.dataBytes,
        sampleRate,
      }), 0, WAV_HEADER_BYTES, 0)
      closeSync(active.fd)
      renameSync(active.partialPath, active.finalPath)
      return {
        state: 'saved',
        recordingId: active.recordingId,
        path: active.finalPath,
        fileName: basename(active.finalPath),
        startedAt: active.startedAt,
        sampleRate,
        durationSeconds: active.dataBytes / (sampleRate * PCM_CHANNELS * 2),
        sourceDurationSeconds: active.sourceBytes
          / ((active.sourceSampleRate || sampleRate) * PCM_CHANNELS * 2),
        speechSegmentCount: active.speechSegmentCount,
      }
    } catch (error) {
      try { closeSync(active.fd) } catch {}
      rmSync(active.partialPath, { force: true })
      throw error
    } finally {
      this.#closeVad()
    }
  }

  discardRecording() {
    const active = this.active
    if (!active) return false
    this.active = null
    try { closeSync(active.fd) } catch {}
    rmSync(active.partialPath, { force: true })
    this.#closeVad()
    return true
  }

  get activeRecordingId() {
    return this.active?.recordingId || ''
  }

  get activePath() {
    return this.active?.finalPath || ''
  }

  #appendSpeechChunks(active, chunks) {
    let bytesWritten = 0
    for (const chunk of Array.isArray(chunks) ? chunks : []) {
      if (!Buffer.isBuffer(chunk) || !chunk.length) continue
      if (chunk.length % 2 !== 0) {
        throw new RangeError('voice reference VAD returned an invalid PCM16 chunk')
      }
      writeSync(active.fd, chunk)
      active.dataBytes += chunk.length
      active.speechSegmentCount += 1
      bytesWritten += chunk.length
    }
    return {
      bytesWritten,
      dataBytes: active.dataBytes,
      sourceBytes: active.sourceBytes,
      speechSegmentCount: active.speechSegmentCount,
    }
  }

  #appendDetectedSpeechSegments(active, segments) {
    const timeline = active.sourceTimeline
    const sourceRate = active.sourceSampleRate
    if (!timeline || !sourceRate) return this.#appendSpeechChunks(active, [])

    let bytesWritten = 0
    for (const segment of Array.isArray(segments) ? segments : []) {
      const startSample = Number(segment?.startSample)
      const endSample = Number(segment?.endSample)
      if (!Number.isFinite(startSample) || !Number.isFinite(endSample) || endSample <= startSample) {
        continue
      }
      const start = Math.max(
        active.lastEmittedSourceSample,
        Math.floor(startSample * sourceRate / VAD_SAMPLE_RATE),
      )
      const end = Math.min(
        active.sourceSamples,
        Math.ceil(endSample * sourceRate / VAD_SAMPLE_RATE),
      )
      if (end <= start) continue
      const chunk = timeline.slice(start, end)
      if (!chunk.length) continue
      writeSync(active.fd, chunk)
      active.dataBytes += chunk.length
      active.speechSegmentCount += 1
      active.lastEmittedSourceSample = end
      bytesWritten += chunk.length
      timeline.discardBefore(end)
    }
    return {
      bytesWritten,
      dataBytes: active.dataBytes,
      sourceBytes: active.sourceBytes,
      speechSegmentCount: active.speechSegmentCount,
    }
  }

  #closeVad() {
    const vad = this.vad
    this.vad = null
    try { vad?.close?.() } catch {}
  }
}

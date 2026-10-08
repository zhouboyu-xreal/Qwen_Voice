import { createRequire } from 'node:module'
import { ensureVoiceReferenceVadModel } from './vad-model-manager.mjs'

const require = createRequire(import.meta.url)
const VAD_SAMPLE_RATE = 16_000

function pcm16ToFloat32(chunk) {
  const samples = new Float32Array(chunk.length / 2)
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = chunk.readInt16LE(index * 2) / 32768
  }
  return samples
}

function float32ToPcm16(samples) {
  const chunk = Buffer.alloc(samples.length * 2)
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]))
    chunk.writeInt16LE(Math.round(value * 32767), index * 2)
  }
  return chunk
}

function concatFloat32(left, right) {
  if (!left.length) return right
  if (!right.length) return left
  const merged = new Float32Array(left.length + right.length)
  merged.set(left)
  merged.set(right, left.length)
  return merged
}

/**
 * Keeps resampling state across renderer audio callbacks. Desktop microphones
 * commonly run at 44.1/48 kHz, while the bundled Silero model is fixed at
 * 16 kHz. Keeping this conversion beside the VAD makes recording safe even
 * when a stale renderer build sends device-rate PCM to the main process.
 */
function createStreamingResampler(outputSampleRate = VAD_SAMPLE_RATE) {
  let inputSampleRate = 0
  let pending = new Float32Array(0)
  let position = 0

  const process = (samples, sampleRate) => {
    const normalizedRate = Number(sampleRate)
    if (!Number.isInteger(normalizedRate) || normalizedRate <= 0) {
      throw new RangeError('voice reference VAD input sample rate is invalid')
    }
    if (inputSampleRate && inputSampleRate !== normalizedRate) {
      throw new RangeError('voice reference VAD input sample rate changed mid-stream')
    }
    inputSampleRate = normalizedRate
    if (normalizedRate === outputSampleRate) return samples

    pending = concatFloat32(pending, samples)
    const step = normalizedRate / outputSampleRate
    const output = []
    while (position + 1 < pending.length) {
      const lower = Math.floor(position)
      const fraction = position - lower
      output.push(pending[lower] + (pending[lower + 1] - pending[lower]) * fraction)
      position += step
    }
    const consumed = Math.floor(position)
    if (consumed > 0) {
      pending = pending.slice(consumed)
      position -= consumed
    }
    return Float32Array.from(output)
  }

  const flush = () => {
    if (!pending.length) return new Float32Array(0)
    const output = []
    while (position < pending.length) {
      const lower = Math.floor(position)
      const upper = Math.min(lower + 1, pending.length - 1)
      const fraction = position - lower
      output.push(pending[lower] + (pending[upper] - pending[lower]) * fraction)
      position += inputSampleRate / outputSampleRate
    }
    pending = new Float32Array(0)
    position = 0
    inputSampleRate = 0
    return Float32Array.from(output)
  }

  return { process, flush }
}

function drainDetectedSpeech(vad) {
  const chunks = []
  while (vad.isDetected()) {
    const segment = vad.front()
    vad.pop()
    if (segment?.samples?.length) chunks.push(float32ToPcm16(segment.samples))
  }
  return chunks
}

/**
 * Creates a local 16 kHz Silero VAD adapter. The adapter returns complete
 * speech segments, allowing the recorder to concatenate only speech into the
 * saved reference WAV without retaining the non-speech source audio.
 */
export async function createVoiceReferenceVad({ modelRoot } = {}) {
  if (!modelRoot) throw new TypeError('voice reference VAD model root is required')
  const model = await ensureVoiceReferenceVadModel(modelRoot)
  const { createVad } = require('sherpa-onnx')
  const vad = createVad({
    sileroVad: {
      model,
      threshold: 0.5,
      minSilenceDuration: 0.45,
      minSpeechDuration: 0.25,
      maxSpeechDuration: 30,
      windowSize: 512,
    },
    sampleRate: VAD_SAMPLE_RATE,
    numThreads: 1,
    provider: 'cpu',
    debug: 0,
    bufferSizeInSeconds: 60,
  })
  const resampler = createStreamingResampler()
  let closed = false
  const accept = (chunk, sampleRate) => {
    if (closed || !Buffer.isBuffer(chunk) || !chunk.length) return []
    const samples = resampler.process(pcm16ToFloat32(chunk), sampleRate)
    if (!samples.length) return []
    vad.acceptWaveform(samples)
    return drainDetectedSpeech(vad)
  }
  return {
    acceptPcm16(chunk, sampleRate) {
      return accept(chunk, sampleRate)
    },
    flush() {
      if (closed) return []
      const remaining = resampler.flush()
      if (remaining.length) vad.acceptWaveform(remaining)
      vad.flush()
      return drainDetectedSpeech(vad)
    },
    close() {
      if (closed) return
      closed = true
      vad.free()
    },
    outputSampleRate: VAD_SAMPLE_RATE,
  }
}

export { VAD_SAMPLE_RATE }

import { pcmBase64 } from '../realtime/audio.js'

/** Dedicated capture graph: its only output is the ambient recording IPC. */
function createDisplaySystemAudioCapture({
  mediaDevices = navigator.mediaDevices,
  createContext = () => new AudioContext(),
  onAudio,
  onEnded,
  now = () => performance.timeOrigin + performance.now(),
}) {
  let stream = null
  let context = null
  let source = null
  let processor = null
  let sink = null
  let parts = []
  let sampleCount = 0
  let chunkStartedAt = 0
  let recording = false
  let closing = false
  let generation = 0

  function flush() {
    if (!sampleCount || !context) return
    const samples = new Float32Array(sampleCount)
    let offset = 0
    for (const part of parts) { samples.set(part, offset); offset += part.length }
    parts = []
    sampleCount = 0
    onAudio(pcmBase64(samples), context.sampleRate, new Date(chunkStartedAt).toISOString())
  }

  function stop() {
    generation += 1
    closing = true
    if (recording) flush()
    recording = false
    if (processor) processor.onaudioprocess = null
    source?.disconnect()
    processor?.disconnect()
    sink?.disconnect()
    stream?.getTracks().forEach(track => track.stop())
    void context?.close().catch(() => {})
    stream = context = source = processor = sink = null
    parts = []
    sampleCount = 0
  }

  return {
    async start() {
      if (!mediaDevices?.getDisplayMedia) throw new Error('当前环境不支持系统音频采集')
      closing = false
      const current = ++generation
      const captured = await mediaDevices.getDisplayMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        video: { width: 320, height: 180, frameRate: 1 },
        systemAudio: 'include',
      })
      if (current !== generation) {
        captured.getTracks().forEach(track => track.stop())
        throw new Error('系统音频采集已取消')
      }
      stream = captured
      const audioTrack = stream.getAudioTracks().find(track => track.readyState === 'live')
      if (!audioTrack) {
        stop()
        throw new Error('未获取到系统音频，请检查录音权限和共享音频选项')
      }
      for (const track of stream.getTracks()) {
        track.addEventListener('ended', () => {
          if (!closing) { stop(); onEnded?.('系统音频采集已中断，环境录音已停止') }
        }, { once: true })
      }
      try {
        context = createContext()
        await context.resume()
        if (current !== generation || audioTrack.readyState !== 'live') throw new Error('系统音频采集已中断')
        source = context.createMediaStreamSource(stream)
        processor = context.createScriptProcessor(2048, 1, 1)
        sink = context.createGain()
        sink.gain.value = 0
        processor.onaudioprocess = event => {
          if (!recording) return
          const samples = new Float32Array(event.inputBuffer.getChannelData(0))
          if (!sampleCount) chunkStartedAt = now() - samples.length / context.sampleRate * 1000
          parts.push(samples)
          sampleCount += samples.length
          if (sampleCount >= context.sampleRate) flush()
        }
        source.connect(processor)
        processor.connect(sink)
        sink.connect(context.destination)
        // Keep API-supplied video tracks alive for capture compatibility.
        // No video frames are read, saved or sent.
      } catch (error) {
        stop()
        throw error
      }
    },
    begin() { recording = true },
    flush,
    stop,
  }
}


function createNativeSystemAudioCapture({ desktop, captureToken, onActivity }) {
  let unsubscribe = null
  let stopping = null
  return {
    async start() {
      unsubscribe = desktop.onAmbientSystemAudioProgress(details => {
        if (details.captureToken === captureToken) onActivity?.()
      })
      try { await desktop.startNativeSystemAudioCapture(captureToken) }
      catch (error) { unsubscribe?.(); unsubscribe = null; throw error }
    },
    begin() {}, // Main starts PCM delivery after the recorder is ready.
    flush() {},
    stop() {
      unsubscribe?.()
      unsubscribe = null
      stopping ||= desktop.stopNativeSystemAudioCapture(captureToken)
      return stopping
    },
  }
}

export function createAmbientSystemAudioCapture(options) {
  return options.backend === 'native'
    ? createNativeSystemAudioCapture(options)
    : createDisplaySystemAudioCapture(options)
}

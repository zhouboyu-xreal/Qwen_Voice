import { createAmbientSystemAudioCapture } from './ambient-system-audio.js'

/** Both tracks share one recording session. Only microphone enters voice/wake. */
export function createAmbientRecordingController({
  desktop, flushMicrophone, onError,
  createSystemCapture = createAmbientSystemAudioCapture,
  now = () => performance.now(),
  setIntervalFn = setInterval,
  clearIntervalFn = clearInterval,
}) {
  let capture = null
  let recordingId = ''
  let generation = 0
  let watchdog = null
  let lastAudio = {}

  function clearWatchdog() {
    if (watchdog !== null) clearIntervalFn(watchdog)
    watchdog = null
  }

  function append(audio, sampleRate, source, startedAt) {
    if (!recordingId) return
    lastAudio[source] = now()
    desktop.appendAmbientRecordingAudio(audio, sampleRate, { source, recordingId, startedAt })
  }

  async function stop(options) {
    generation += 1
    clearWatchdog()
    flushMicrophone()
    try { await capture?.stop() }
    catch (error) { options = { ...options, captureError: options?.captureError || error.message } }
    finally { capture = null }
    flushMicrophone()
    recordingId = ''
    return desktop.stopAmbientRecording(options)
  }

  return {
    async start() {
      const current = ++generation
      try {
        const { sources, systemCapture, captureToken } = await desktop.prepareAmbientRecording()
        if (current !== generation) throw new Error('环境录音启动已取消')
        if (sources.includes('system')) {
          capture = createSystemCapture({
            desktop, backend: systemCapture, captureToken,
            onActivity: () => { if (recordingId) lastAudio.system = now() },
            onAudio: (audio, sampleRate, startedAt) => append(audio, sampleRate, 'system', startedAt),
            onEnded: message => {
              onError(message)
              void stop({ captureError: message }).catch(error => onError(error.message))
            },
          })
          await capture.start()
        }
        if (current !== generation) throw new Error('环境录音启动已取消')
        const result = await desktop.startAmbientRecording()
        if (current !== generation) throw new Error('环境录音启动已取消')
        recordingId = result.recordingId
        capture?.begin()
        lastAudio = Object.fromEntries(sources.map(source => [source, now()]))
        watchdog = setIntervalFn(() => {
          const stale = sources.find(source => now() - lastAudio[source] > 10000)
          if (!stale) return
          const message = stale === 'system'
            ? '系统音频采集中断，环境录音已停止'
            : '麦克风音频采集中断，环境录音已停止'
          onError(message)
          void stop({ captureError: message }).catch(error => onError(error.message))
        }, 2000)
        return result
      } catch (error) {
        clearWatchdog()
        await Promise.resolve(capture?.stop()).catch(() => {})
        capture = null
        recordingId = ''
        // Also revoke a prepared capture permission after picker cancellation.
        await desktop.stopAmbientRecording().catch(() => {})
        throw error
      }
    },
    stop,
    appendMicrophone: (audio, sampleRate, startedAt) => append(audio, sampleRate, 'microphone', startedAt),
    failMicrophone(message) {
      if (!recordingId) return
      onError(message)
      void stop({ captureError: message }).catch(error => onError(error.message))
    },
    release() {
      generation += 1
      clearWatchdog()
      void Promise.resolve(capture?.stop()).catch(() => {})
      capture = null
      recordingId = ''
    },
  }
}

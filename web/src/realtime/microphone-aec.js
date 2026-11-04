const MODES = ['default', 'all']
const FILTERS = ['echoCancellation', 'noiseSuppression', 'autoGainControl']
const NUMBERS = ['sampleRate', 'sampleSize', 'channelCount', 'latency']
const echoValue = value => typeof value === 'boolean' || ['all', 'remote-only'].includes(value)
const text = value => typeof value === 'string' ? value.slice(0, 160) : ''
function read(call) { try { return call() || {} } catch { return {} } }

export function microphoneCaptureDiagnostics(track, mediaDevices) {
  const actual = read(() => track.getSettings())
  const available = read(() => track.getCapabilities())
  const supported = read(() => mediaDevices.getSupportedConstraints())
  const settings = {}
  const capabilities = {}
  const supportedConstraints = {}
  for (const key of FILTERS) {
    if (key === 'echoCancellation' ? echoValue(actual[key]) : typeof actual[key] === 'boolean') settings[key] = actual[key]
    if (Array.isArray(available[key])) capabilities[key] = available[key]
      .filter(value => key === 'echoCancellation' ? echoValue(value) : typeof value === 'boolean').slice(0, 4)
  }
  for (const key of NUMBERS) {
    if (Number.isFinite(actual[key])) settings[key] = actual[key]
    if (available[key] && Number.isFinite(available[key].min) && Number.isFinite(available[key].max)) {
      capabilities[key] = { min: available[key].min, max: available[key].max }
    }
  }
  for (const key of [...FILTERS, ...NUMBERS]) {
    if (typeof supported[key] === 'boolean') supportedConstraints[key] = supported[key]
  }
  // Device/group identifiers and arbitrary track properties never enter logs.
  return { device: { label: text(track?.label) }, settings, capabilities, supportedConstraints }
}

/** Strict all-mode negotiation; only a reported "all" counts as confirmation. */
export async function acquireMicrophoneWithDiagnostics({
  mediaDevices = navigator.mediaDevices,
  mode = 'default',
  deviceId = '',
  verifyFrames = null,
  onDiagnostics = () => {},
} = {}) {
  if (!MODES.includes(mode)) throw new TypeError('Invalid microphone echo cancellation mode')
  const captureId = crypto.randomUUID()
  const notify = details => {
    try { onDiagnostics({ captureId, requestedMode: mode, ...details }) } catch { /* Diagnostics cannot interrupt audio. */ }
  }
  const acquire = echoCancellation => mediaDevices.getUserMedia({ audio: {
    echoCancellation, noiseSuppression: true, autoGainControl: true,
    ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
  } })
  let media
  let allSupport = 'not-tested'
  let fallbackReason = ''
  let negotiationError = ''
  let allPcmFlow = 'not-tested'
  let allNegotiatedEchoCancellation
  try {
    if (mode === 'all') {
      try {
        media = await acquire({ exact: 'all' })
        const track = media.getAudioTracks()[0]
        allNegotiatedEchoCancellation = read(() => track.getSettings()).echoCancellation
        if (allNegotiatedEchoCancellation === 'all') {
          allSupport = 'confirmed'
          if (verifyFrames) {
            allPcmFlow = await verifyFrames(media) ? 'received' : 'timeout'
            if (allPcmFlow === 'timeout') {
              allSupport = 'no-pcm'
              fallbackReason = 'all-no-pcm'
              media.getTracks().forEach(track => track.stop())
              media = null
            }
          }
        } else {
          allSupport = 'unconfirmed'
          fallbackReason = 'all-not-reported'
          media.getTracks().forEach(track => track.stop())
          media = null
        }
      } catch (error) {
        const unsupported = (error.name === 'OverconstrainedError'
          && (!error.constraint || error.constraint === 'echoCancellation'))
          || ['NotSupportedError', 'TypeError'].includes(error.name)
        if (!unsupported) throw error
        allSupport = 'rejected'
        fallbackReason = 'all-constraint-rejected'
        negotiationError = text(error.name)
      }
    }
    if (!media) {
      media = await acquire(true)
      if (fallbackReason === 'all-no-pcm' && verifyFrames && !await verifyFrames(media)) {
        throw Object.assign(new Error('Microphone stopped delivering audio frames'), { name: 'NotReadableError' })
      }
    }
    const track = media.getAudioTracks()[0]
    if (!track) throw Object.assign(new Error('No microphone audio track'), { name: 'NotFoundError' })
    const snapshot = microphoneCaptureDiagnostics(track, mediaDevices)
    if (snapshot.settings.echoCancellation === 'all' && allSupport !== 'no-pcm') allSupport = 'confirmed'
    else if (mode === 'default' && snapshot.capabilities.echoCancellation?.includes('all')) allSupport = 'advertised'
    const diagnostic = {
      phase: 'opened', appliedMode: allSupport === 'confirmed' ? 'all' : 'default',
      allSupport, fallbackReason, negotiationError, allPcmFlow, allNegotiatedEchoCancellation, ...snapshot,
    }
    notify(diagnostic)
    return { media, diagnostics: { captureId, requestedMode: mode, ...diagnostic } }
  } catch (error) {
    media?.getTracks().forEach(track => track.stop())
    notify({ phase: 'failed', allSupport, fallbackReason, negotiationError, allPcmFlow, allNegotiatedEchoCancellation, errorName: text(error.name) })
    throw error
  }
}

/** Silence still produces frames. A stalled source must never count as clean AEC. */
export function waitForMicrophoneFrames({ media, context, timeoutMs = 3000 }) {
  return new Promise(resolve => {
    let source
    let processor
    let sink
    let timer
    let settled = false
    const finish = received => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      if (processor) processor.onaudioprocess = null
      source?.disconnect()
      processor?.disconnect()
      sink?.disconnect()
      resolve(received)
    }
    try {
      source = context.createMediaStreamSource(media)
      processor = context.createScriptProcessor(2048, 1, 1)
      sink = context.createGain()
      sink.gain.value = 0
      processor.onaudioprocess = event => {
        if (event.inputBuffer.length > 0) finish(true)
      }
      timer = setTimeout(() => finish(false), timeoutMs)
      source.connect(processor)
      processor.connect(sink)
      sink.connect(context.destination)
    } catch { finish(false) }
  })
}

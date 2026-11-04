import { mkdirSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
const text = (value, limit = 160) => typeof value === 'string' ? value.replace(/[\r\n\0]/g, ' ').slice(0, limit) : ''
const pick = (value, allowed, fallback = '') => allowed.includes(value) ? value : fallback
const echoValue = value => typeof value === 'boolean' || ['all', 'remote-only'].includes(value)

export function sanitizeMicrophoneDiagnostics(payload) {
  if (!payload || !['opened', 'failed'].includes(payload.phase)) return null
  const safe = {
    captureId: text(payload.captureId, 80),
    requestedMode: pick(payload.requestedMode, ['default', 'all'], 'default'),
    phase: payload.phase,
    reason: pick(payload.reason, ['initial', 'devicechange', 'retry', 'track-muted', 'track-ended', 'app-resume']),
    appliedMode: pick(payload.appliedMode, ['default', 'all']),
    allSupport: pick(payload.allSupport, ['not-tested', 'advertised', 'confirmed', 'rejected', 'unconfirmed', 'no-pcm']),
    fallbackReason: pick(payload.fallbackReason, ['all-not-reported', 'all-constraint-rejected', 'all-no-pcm']),
    allPcmFlow: pick(payload.allPcmFlow, ['not-tested', 'received', 'timeout']),
    ...(echoValue(payload.allNegotiatedEchoCancellation) ? { allNegotiatedEchoCancellation: payload.allNegotiatedEchoCancellation } : {}),
    negotiationError: pick(payload.negotiationError, ['OverconstrainedError', 'NotSupportedError', 'TypeError']),
    errorName: pick(payload.errorName, ['NotAllowedError', 'SecurityError', 'NotFoundError', 'NotReadableError', 'NotSupportedError', 'OverconstrainedError', 'TypeError', 'AbortError']),
    device: { label: text(payload.device?.label) },
    settings: {}, capabilities: {}, supportedConstraints: {},
  }
  for (const key of ['echoCancellation', 'noiseSuppression', 'autoGainControl']) {
    const accept = value => key === 'echoCancellation' ? echoValue(value) : typeof value === 'boolean'
    if (accept(payload.settings?.[key])) safe.settings[key] = payload.settings[key]
    if (Array.isArray(payload.capabilities?.[key])) safe.capabilities[key] = payload.capabilities[key].filter(accept).slice(0, 4)
  }
  for (const key of ['sampleRate', 'sampleSize', 'channelCount', 'latency']) {
    if (Number.isFinite(payload.settings?.[key])) safe.settings[key] = payload.settings[key]
    const range = payload.capabilities?.[key]
    if (Number.isFinite(range?.min) && Number.isFinite(range?.max)) safe.capabilities[key] = { min: range.min, max: range.max }
  }
  for (const key of ['echoCancellation', 'noiseSuppression', 'autoGainControl', 'sampleRate', 'sampleSize', 'channelCount', 'latency']) {
    if (typeof payload.supportedConstraints?.[key] === 'boolean') safe.supportedConstraints[key] = payload.supportedConstraints[key]
  }
  if (Number.isFinite(payload.contextSampleRate)) safe.contextSampleRate = payload.contextSampleRate
  if (payload.phase === 'opened' && safe.settings.echoCancellation !== 'all') {
    if (safe.allSupport === 'confirmed') safe.allSupport = 'unconfirmed'
    if (safe.appliedMode === 'all') safe.appliedMode = 'default'
  }
  return safe
}

export function createMicrophoneDiagnosticStore({ logger, runtime = {} } = {}) {
  let latest = null
  let active = null
  function persist() {
    if (!active) return
    try {
      mkdirSync(active.directory, { recursive: true, mode: 0o700 })
      const path = join(active.directory, `${active.recordingId}.microphone.json`)
      const { directory: _directory, ...document } = active
      writeFileSync(path + '.tmp', JSON.stringify(document, null, 2) + '\n', { mode: 0o600 })
      renameSync(path + '.tmp', path)
    } catch (error) {
      logger?.warn('desktop.microphone.diagnostics_write_failed', { code: error.code })
    }
  }
  return {
    record(payload) {
      const safe = sanitizeMicrophoneDiagnostics(payload)
      if (!safe) return
      latest = { ...safe, observedAt: new Date().toISOString() }
      logger?.info('desktop.microphone.capture', { ...latest, recordingId: active?.recordingId || null })
      if (active) { active.captures = [...active.captures, latest].slice(-20); persist() }
    },
    begin({ recordingId, directory, requestedMode }) {
      if (!/^[A-Za-z0-9_-]{1,128}$/.test(recordingId)) throw new TypeError('Invalid recording id')
      active = { recordingId, directory, requestedMode, runtime, state: 'recording',
        captures: latest && latest.requestedMode === requestedMode ? [latest] : [] }
      persist()
    },
    end(recordingId) {
      if (active?.recordingId !== recordingId) return
      active.state = 'stopped'
      persist()
      active = null
    },
  }
}

import { spawn } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import {
  resolveAgentMemoryTranscriptIpcSocketPath,
} from './agent-memory-ipc-path.mjs'

const DEFAULT_TIMEOUT_MS = 30_000
const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const BUNDLED_MEMORY_DIRECTORY = join(PROJECT_ROOT, 'memory')
const BUNDLED_SIDECAR = join(
  BUNDLED_MEMORY_DIRECTORY,
  'integrations/qwen_audio_agent/ambient_audio_recording_sidecar.py',
)
const BUNDLED_MEMORY_CONFIG = join(BUNDLED_MEMORY_DIRECTORY, 'config.yaml')

function clean(value, limit = 2_000) {
  return [...String(value || '').replaceAll('\0', '').trim()]
    .slice(0, limit)
    .join('')
}

function defaultPythonCommand(env) {
  if (String(env.AGENT_MEMORY_PYTHON || '').trim()) return env.AGENT_MEMORY_PYTHON
  return process.platform === 'win32' ? 'python' : 'python3'
}

function defaultAgentMemoryOwnerId(env) {
  const explicit = String(env.AGENT_MEMORY_LOCAL_OWNER_ID || '').trim()
  if (explicit) return explicit
  const owner = String(env.QWEN_AUDIO_AGENT_PERSONAL_OWNER_ID || 'user_personal')
  return createHash('sha256').update(owner).digest('hex')
}

export function ambientAudioRecordingEnabled(env = process.env) {
  return !['0', 'false', 'no', 'off'].includes(
    String(env.AGENT_MEMORY_AMBIENT_RECORDING_ENABLED || '').trim().toLowerCase(),
  )
}

export function resolveAmbientAudioRecordingSidecarOptions(env = process.env, {
  recordingsDirectory = '',
} = {}) {
  const configuredSidecar = String(
    env.AGENT_MEMORY_AMBIENT_RECORDING_SIDECAR || '',
  ).trim()
  const memorySidecar = String(env.AGENT_MEMORY_SIDECAR || '').trim()
  const sidecarPath = configuredSidecar
    ? resolve(configuredSidecar)
    : memorySidecar
      ? resolve(dirname(resolve(memorySidecar)), 'ambient_audio_recording_sidecar.py')
      : BUNDLED_SIDECAR
  const configPath = String(env.AGENT_MEMORY_CONFIG || '').trim()
  const stateDirectory = String(env.AGENT_MEMORY_STATE_DIR || '').trim()
  const recordingRoot = String(recordingsDirectory || '').trim()
    || (stateDirectory ? join(resolve(stateDirectory), 'ambient-recordings') : '')
  if (!recordingRoot) {
    throw new Error('环境录制需要显式的 recordingsDirectory 或 AGENT_MEMORY_STATE_DIR')
  }
  const configuredIpcSocket = String(env.AGENT_MEMORY_IPC_SOCKET || '').trim()
  const configuredChunkSeconds = Number(env.AGENT_MEMORY_AMBIENT_RECORDING_CHUNK_SECONDS)
  return {
    command: defaultPythonCommand(env),
    sidecarPath,
    configPath: configPath ? resolve(configPath) : BUNDLED_MEMORY_CONFIG,
    recordingsDirectory: resolve(recordingRoot),
    agentMemoryIpcSocket: resolveAgentMemoryTranscriptIpcSocketPath({
      stateDirectory: stateDirectory || dirname(resolve(recordingRoot)),
      configuredPath: configuredIpcSocket,
    }),
    agentMemoryOwnerId: defaultAgentMemoryOwnerId(env),
    chunkDurationSeconds: Number.isFinite(configuredChunkSeconds)
      && configuredChunkSeconds > 0
      ? configuredChunkSeconds
      : 600,
    logPath: stateDirectory
      ? join(resolve(stateDirectory), 'ambient-audio-recording-sidecar.log')
      : '',
    env: { ...env },
  }
}

/**
 * JSONL adapter for the durable ambient recording sidecar.
 *
 * The adapter never writes memory itself. The child sidecar sends completed
 * ASR batches directly to the primary Agent Memory sidecar over its private
 * local Unix socket; the Gateway is not on this data path.
 */
export class AmbientAudioRecordingRuntime {
  constructor({
    command,
    sidecarPath,
    configPath = '',
    recordingsDirectory,
    agentMemoryIpcSocket,
    agentMemoryOwnerId,
    chunkDurationSeconds = 600,
    logPath = '',
    env = process.env,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = {}) {
    this.command = command || defaultPythonCommand(env)
    this.sidecarPath = String(sidecarPath || '')
    this.configPath = String(configPath || '')
    this.recordingsDirectory = String(recordingsDirectory || '')
    this.agentMemoryIpcSocket = String(agentMemoryIpcSocket || '')
    this.agentMemoryOwnerId = String(agentMemoryOwnerId || '')
    this.chunkDurationSeconds = Number(chunkDurationSeconds) || 600
    this.logPath = String(logPath || '')
    this.env = { ...env }
    this.timeoutMs = timeoutMs
    this.child = null
    this.pending = new Map()
    this.pendingAudioRequests = 0
    this.lastError = ''
    this.ready = false
  }

  validate() {
    if (!this.sidecarPath || !existsSync(this.sidecarPath)) {
      throw new Error(`环境录制 sidecar 不存在：${this.sidecarPath}`)
    }
    if (!this.recordingsDirectory) {
      throw new Error('环境录制目录未配置')
    }
    if (!this.agentMemoryIpcSocket || !this.agentMemoryOwnerId) {
      throw new Error('Agent Memory 本地 IPC 未配置')
    }
  }

  start() {
    if (this.child) return
    this.validate()
    this.lastError = ''
    this.child = spawn(this.command, [
      this.sidecarPath,
      '--recordings-dir', this.recordingsDirectory,
      '--chunk-duration-seconds', String(this.chunkDurationSeconds),
      '--agent-memory-ipc-socket', this.agentMemoryIpcSocket,
      '--agent-memory-owner-id', this.agentMemoryOwnerId,
      ...(this.configPath ? ['--config', this.configPath] : []),
      ...(this.logPath ? ['--log-path', this.logPath] : []),
    ], {
      cwd: dirname(this.sidecarPath),
      env: this.env,
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    createInterface({ input: this.child.stdout }).on('line', line => {
      let message
      try { message = JSON.parse(line) } catch { return }
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      clearTimeout(pending.timer)
      if (pending.audio) this.pendingAudioRequests -= 1
      if (message.error) pending.reject(new Error(message.error))
      else pending.resolve(message.result)
    })
    this.child.stderr.on('data', chunk => {
      this.lastError = clean(chunk, 500)
    })
    this.child.once('error', error => {
      this.lastError = clean(error.message, 500)
    })
    this.child.once('exit', (code, signal) => {
      const error = new Error(
        this.lastError || `ambient recording sidecar exited (${signal || code})`,
      )
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer)
        pending.reject(error)
      }
      this.pending.clear()
      this.pendingAudioRequests = 0
      this.ready = false
      this.child = null
    })
  }

  request(method, params = {}, {
    timeoutMs = this.timeoutMs,
    audio = false,
  } = {}) {
    this.start()
    const id = randomUUID()
    return new Promise((resolveRequest, rejectRequest) => {
      const timer = setTimeout(() => {
        const pending = this.pending.get(id)
        this.pending.delete(id)
        if (pending?.audio) this.pendingAudioRequests -= 1
        rejectRequest(new Error(`ambient recording ${method} timed out`))
      }, timeoutMs)
      timer.unref?.()
      this.pending.set(id, {
        resolve: resolveRequest,
        reject: rejectRequest,
        timer,
        audio,
      })
      this.child.stdin.write(`${JSON.stringify({ id, method, params })}\n`, error => {
        if (!error) return
        const pending = this.pending.get(id)
        clearTimeout(timer)
        this.pending.delete(id)
        if (pending?.audio) this.pendingAudioRequests -= 1
        rejectRequest(error)
      })
    })
  }

  async startRecording({ recordingId, startedAt } = {}) {
    const result = await this.request('recording.start', {
      recordingId: clean(recordingId, 128),
      startedAt: clean(startedAt, 120),
    }, { timeoutMs: 120_000 })
    this.ready = true
    return result
  }

  appendPcm16(chunk, sampleRate, recordingId) {
    if (
      !this.ready
      || !Buffer.isBuffer(chunk)
      || !chunk.length
    ) return null
    this.pendingAudioRequests += 1
    return this.request('audio.append', {
      recordingId: clean(recordingId, 128),
      audio: chunk.toString('base64'),
      sampleRate,
    }, { timeoutMs: 5_000, audio: true })
  }

  drain(recordingId) {
    if (!this.ready) return Promise.resolve(null)
    return this.request('recording.drain', {
      recordingId: clean(recordingId, 128),
    }, { timeoutMs: 5_000 })
  }

  stopRecording(recordingId) {
    if (!this.ready) return Promise.resolve(null)
    return this.request('recording.stop', {
      recordingId: clean(recordingId, 128),
    }, { timeoutMs: 10_000 })
  }

  async close() {
    if (!this.child) return
    try { await this.request('close', {}, { timeoutMs: 2_000 }) } catch {}
    this.child?.kill()
    this.child = null
    this.ready = false
  }
}

import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'

const DEFAULT_TIMEOUT_MS = 30_000
const MAX_PENDING_AUDIO_REQUESTS = 4
const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const BUNDLED_MEMORY_DIRECTORY = join(PROJECT_ROOT, 'memory')
const BUNDLED_PREWAKE_SIDECAR = join(
  BUNDLED_MEMORY_DIRECTORY,
  'integrations/qwen_audio_agent/prewake_context_sidecar.py',
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

export function preWakeContextEnabled(env = process.env) {
  return ['1', 'true', 'yes', 'on'].includes(
    String(env.AGENT_MEMORY_PREWAKE_CONTEXT_ENABLED || '').trim().toLowerCase(),
  )
}

export function resolvePreWakeContextSidecarOptions(env = process.env) {
  const configuredSidecar = String(env.AGENT_MEMORY_PREWAKE_CONTEXT_SIDECAR || '').trim()
  const memorySidecar = String(env.AGENT_MEMORY_SIDECAR || '').trim()
  const sidecarPath = configuredSidecar
    ? resolve(configuredSidecar)
    : memorySidecar
      ? resolve(dirname(resolve(memorySidecar)), 'prewake_context_sidecar.py')
      : BUNDLED_PREWAKE_SIDECAR
  const configPath = String(env.AGENT_MEMORY_CONFIG || '').trim()
  const stateDirectory = String(env.AGENT_MEMORY_STATE_DIR || '').trim()
  return {
    command: defaultPythonCommand(env),
    sidecarPath,
    configPath: configPath ? resolve(configPath) : BUNDLED_MEMORY_CONFIG,
    logPath: stateDirectory
      ? join(resolve(stateDirectory), 'prewake-context-sidecar.log')
      : '',
    env: { ...env },
  }
}

/**
 * JSONL adapter for agent_memory's local pre-wake ASR runtime.
 *
 * TUI and Desktop deliberately share this adapter: their audio capture differs,
 * but buffering, local ASR, snapshotting and failure semantics must not.
 */
export class PreWakeContextRuntime {
  constructor({
    command,
    sidecarPath,
    configPath = '',
    logPath = '',
    env = process.env,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = {}) {
    this.command = command || defaultPythonCommand(env)
    this.sidecarPath = String(sidecarPath || '')
    this.configPath = String(configPath || '')
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
    if (!this.sidecarPath) {
      throw new Error('未配置 Agent Memory pre-wake sidecar')
    }
    if (!existsSync(this.sidecarPath)) {
      throw new Error(`pre-wake sidecar 不存在：${this.sidecarPath}`)
    }
  }

  start() {
    if (this.child) return
    this.validate()
    this.lastError = ''
    this.child = spawn(this.command, [
      this.sidecarPath,
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
        this.lastError || `pre-wake sidecar exited (${signal || code})`,
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
        rejectRequest(new Error(`pre-wake ${method} timed out`))
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

  async startWhenReady() {
    await this.request('health', {}, { timeoutMs: 120_000 })
    this.ready = true
    return true
  }

  appendPcm16(chunk, sampleRate) {
    if (!this.ready || !Buffer.isBuffer(chunk) || !chunk.length) return false
    if (this.pendingAudioRequests >= MAX_PENDING_AUDIO_REQUESTS) return false
    this.pendingAudioRequests += 1
    try {
      this.request('audio.append', {
        audio: chunk.toString('base64'),
        sampleRate,
        endedAt: Date.now() / 1_000,
      }, { timeoutMs: 2_000, audio: true }).catch(error => {
        this.lastError = clean(error.message, 500)
      })
    } catch (error) {
      this.pendingAudioRequests -= 1
      this.lastError = clean(error.message, 500)
      return false
    }
    return true
  }

  async consumeForWake({ timeoutMs = 350 } = {}) {
    if (!this.ready) return null
    const snapshot = await this.request('wake.snapshot', {}, { timeoutMs })
    const text = clean(snapshot?.text, 1_200)
    return text ? { text } : null
  }

  clear() {
    if (!this.ready) return
    try {
      this.request('clear', {}, { timeoutMs: 1_000 }).catch(() => {})
    } catch {
      // The wake-word path must remain usable if the optional sidecar exits.
    }
  }

  async close() {
    if (!this.child) return
    try { await this.request('close', {}, { timeoutMs: 1_000 }) } catch {}
    this.child?.kill()
    this.child = null
    this.ready = false
  }
}

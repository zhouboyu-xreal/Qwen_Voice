import { spawn } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import {
  MEMORY_PROVIDER_PROTOCOL_VERSION,
} from '../../provider.mjs'

const SENSITIVE = /(?:api[_ -]?key|secret|token|password|passwd|credential|密码|密钥|验证码|令牌|证件号|身份证|详细住址|病史|病历|诊断|用药|\bsk-[a-z0-9_-]+|\b\d{11,19}\b)/iu

function ownerKey(ownerId) {
  return createHash('sha256').update(String(ownerId || 'anonymous')).digest('hex')
}

function clean(value, limit = 8_000) {
  return [...String(value || '').replaceAll('\0', '').trim()].slice(0, limit).join('')
}

function defaultPythonCommand(env) {
  if (String(env.AGENT_MEMORY_PYTHON || '').trim()) return env.AGENT_MEMORY_PYTHON
  return process.platform === 'win32' ? 'python' : 'python3'
}

class JsonLineSidecar {
  constructor({ command, args, cwd, env, timeoutMs = 30_000 }) {
    this.command = command
    this.args = args
    this.cwd = cwd
    this.env = env
    this.timeoutMs = timeoutMs
    this.child = null
    this.pending = new Map()
    this.lastError = null
    this.lastStderr = null
  }

  start() {
    if (this.child) return
    this.lastError = null
    this.child = spawn(this.command, this.args, {
      cwd: this.cwd,
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
      if (message.error) pending.reject(new Error(message.error))
      else pending.resolve(message.result)
    })
    this.child.stderr.on('data', chunk => {
      this.lastStderr = clean(chunk, 500)
    })
    this.child.once('error', error => {
      this.lastError = clean(error.message, 500)
    })
    this.child.once('exit', (code, signal) => {
      const error = new Error(
        this.lastError || this.lastStderr || `Agent Memory sidecar exited (${signal || code})`,
      )
      if (code && !this.lastError) this.lastError = clean(error.message, 500)
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer)
        pending.reject(error)
      }
      this.pending.clear()
      this.child = null
    })
  }

  request(method, params = {}, { timeoutMs = this.timeoutMs } = {}) {
    this.start()
    const id = randomUUID()
    return new Promise((resolveRequest, rejectRequest) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        rejectRequest(new Error(`Agent Memory ${method} timed out`))
      }, timeoutMs)
      timer.unref?.()
      this.pending.set(id, { resolve: resolveRequest, reject: rejectRequest, timer })
      this.child.stdin.write(`${JSON.stringify({ id, method, params })}\n`, error => {
        if (!error) return
        clearTimeout(timer)
        this.pending.delete(id)
        rejectRequest(error)
      })
    })
  }

  async close({ timeoutMs = this.timeoutMs } = {}) {
    if (!this.child) return
    try { await this.request('close', {}, { timeoutMs }) } catch {}
    this.child?.kill()
    this.child = null
  }
}

/**
 * Optional bridge to the agent_memory Python runtime.
 *
 * Agent Memory is the sole persistence and recall authority. This adapter
 * deliberately exposes no local document snapshot: a synchronous Markdown
 * profile would bypass the Python sidecar and leak stale facts into every new
 * Realtime session through its system instructions.
 */
export class AgentMemoryProvider {
  constructor({
    stateDirectory = resolve(process.cwd(), '.qwen-audio', 'agent-memory'),
    python = null,
    sidecarPath = null,
    configPath = null,
    timeoutMs = 30_000,
    backgroundTimeoutMs = 120_000,
    env = process.env,
    sidecar = null,
  } = {}) {
    this.stateDirectory = resolve(stateDirectory)
    this.backgroundTimeoutMs = Math.max(timeoutMs, backgroundTimeoutMs)
    this.pendingObservations = new Map()
    const sidecarEnvironment = { ...env }
    const configuredSidecar = String(
      sidecarPath || sidecarEnvironment.AGENT_MEMORY_SIDECAR || '',
    ).trim()
    if (!sidecar && !configuredSidecar) {
      throw new Error(
        'Agent Memory 需要外部 sidecar；请设置 AGENT_MEMORY_SIDECAR 为其绝对路径',
      )
    }
    const resolvedSidecar = configuredSidecar ? resolve(configuredSidecar) : ''
    if (!sidecar && !existsSync(resolvedSidecar)) {
      throw new Error(`Agent Memory sidecar 不存在：${resolvedSidecar}`)
    }
    const configuredConfig = String(
      configPath || sidecarEnvironment.AGENT_MEMORY_CONFIG || '',
    ).trim()
    this.sidecar = sidecar || new JsonLineSidecar({
      command: python || defaultPythonCommand(sidecarEnvironment),
      args: [
        resolvedSidecar,
        '--state-dir', this.stateDirectory,
        ...(configuredConfig ? ['--config', resolve(configuredConfig)] : []),
      ],
      cwd: dirname(resolvedSidecar),
      env: sidecarEnvironment,
      timeoutMs,
    })
  }

  describe() {
    return {
      protocolVersion: MEMORY_PROVIDER_PROTOCOL_VERSION,
      key: 'agent-memory',
      label: 'Agent Memory',
      capabilities: {
        semanticQuery: true,
        sessionObservation: true,
        audioStreamObservation: false,
      },
    }
  }

  list() {
    return []
  }

  apply() {
    throw new Error(
      'Agent Memory 不支持本地 memory 文档编辑；请使用 Python sidecar 的记忆流程。',
    )
  }

  async query(ownerId, query, { limit = 5 } = {}, context = {}) {
    const result = await this.sidecar.request('recall', {
      ownerId: ownerKey(ownerId),
      query: clean(query, 2_000),
      topK: Math.max(1, Math.min(10, Number(limit) || 5)),
      timeEnd: clean(context?.timeEnd, 120),
      promptLanguage: clean(context?.promptLanguage, 16),
    })
    return {
      memories: [],
      context: clean(result?.memory_context, 8_000),
    }
  }

  async observe(ownerId, exchange, context = {}) {
    const messages = Array.isArray(exchange?.messages)
      ? exchange.messages.map(message => ({
          id: clean(message?.id, 240),
          role: message?.role === 'assistant' ? 'assistant' : 'user',
          content: clean(message?.content, 8_000),
          turnId: clean(message?.turnId, 240),
          createdAt: Number.isFinite(Number(message?.createdAt))
            ? Number(message.createdAt)
            : null,
        })).filter(message => message.content && !SENSITIVE.test(message.content))
      : []
    if (!messages.length) return { observed: false }
    const owner = ownerKey(ownerId)
    const observationKey = createHash('sha256').update(JSON.stringify({
      ownerId: owner,
      sessionId: clean(context.sessionId, 200),
      messages,
    })).digest('hex')
    const pending = this.pendingObservations.get(observationKey)
    if (pending) return pending
    const operation = this.sidecar.request('observe', {
      ownerId: owner,
      sessionId: clean(context.sessionId, 200),
      messages,
    }, { timeoutMs: this.backgroundTimeoutMs }).finally(() => {
      this.pendingObservations.delete(observationKey)
    })
    this.pendingObservations.set(observationKey, operation)
    return operation
  }

  async flush(ownerId, context = {}) {
    return this.sidecar.request('flush', {
      ownerId: ownerKey(ownerId),
      sessionId: clean(context.sessionId, 200),
    }, { timeoutMs: this.backgroundTimeoutMs })
  }

  health() {
    return {
      ok: !this.sidecar.lastError,
      ...(this.sidecar.lastError ? { warning: this.sidecar.lastError } : {}),
    }
  }

  async close() {
    await this.sidecar.close({ timeoutMs: this.backgroundTimeoutMs })
  }
}

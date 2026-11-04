import { randomUUID } from 'node:crypto'
import {
  chmodSync, constants, copyFileSync, existsSync, readFileSync, statSync,
  unlinkSync, writeFileSync,
} from 'node:fs'
import { dirname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseEnv } from 'node:util'
import { replaceFileSync, withFileTransaction } from './file-transaction-lock.mjs'
import { resolveRuntimePaths } from './path-policy.mjs'

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const CODE_PATHS = {
  AGENT_MEMORY_SIDECAR: 'integrations/qwen_audio_agent/agent_memory_sidecar.py',
  AGENT_MEMORY_CONFIG: 'config.yaml',
  AGENT_MEMORY_PREWAKE_CONTEXT_SIDECAR: 'integrations/qwen_audio_agent/prewake_context_sidecar.py',
  AGENT_MEMORY_AMBIENT_RECORDING_SIDECAR: 'integrations/qwen_audio_agent/ambient_audio_recording_sidecar.py',
}
const STATE_KEY = 'AGENT_MEMORY_STATE_DIR'
const clean = value => String(value || '').trim()
const automatic = value => !value || value.toLowerCase() === 'auto'
function isFile(path) {
  try { return statSync(path).isFile() } catch { return false }
}

// Only repair the known vendored layout after its entire checkout disappeared.
// A typo in a custom file, or a missing file in an existing checkout, must still
// surface rather than silently selecting a different runtime/configuration.
function deletedCheckout(path, suffix) {
  const tail = sep + suffix.split('/').join(sep)
  return path.endsWith(tail) && !existsSync(path.slice(0, -tail.length))
}

export function resolveAgentMemoryEnvironment({
  env = process.env,
  root = env.QWEN_AUDIO_AGENT_RUNTIME_ROOT || SOURCE_ROOT,
  dataDirectory = resolveRuntimePaths({ env, baseDirectory: root }).dataDirectory,
} = {}) {
  const environment = { ...env }
  const automaticKeys = []
  const repairedKeys = []
  const memoryRoot = resolve(root, 'memory')
  const enabled = clean(env.QWEN_AUDIO_MEMORY_PROVIDER) === 'agent-memory'
    || Object.keys(env).some(key => key.startsWith('AGENT_MEMORY_') && clean(env[key]))
    || isFile(join(memoryRoot, CODE_PATHS.AGENT_MEMORY_SIDECAR))
  if (!enabled) return { environment, automaticKeys, repairedKeys }

  for (const [key, relative] of Object.entries(CODE_PATHS)) {
    const value = clean(env[key])
    const auxiliary = key !== 'AGENT_MEMORY_SIDECAR' && key !== 'AGENT_MEMORY_CONFIG'
    if (auxiliary && !value) continue // Auxiliary sidecars follow the primary sidecar.
    const bundled = join(memoryRoot, relative)
    const fallback = auxiliary && environment.AGENT_MEMORY_SIDECAR
      ? join(dirname(environment.AGENT_MEMORY_SIDECAR), relative.split('/').at(-1))
      : bundled
    const configured = automatic(value) ? '' : resolve(root, value)
    const repair = configured && !existsSync(configured) && isFile(fallback)
      && deletedCheckout(configured, `memory/${relative}`)
    if (!configured || configured === fallback || repair) {
      environment[key] = fallback
      automaticKeys.push(key)
      if (repair) repairedKeys.push(key)
    } else {
      environment[key] = configured
    }
  }

  const state = clean(env[STATE_KEY])
  const configuredState = automatic(state) ? '' : resolve(root, state)
  const repairState = configuredState && !existsSync(configuredState)
    && deletedCheckout(configuredState, 'tmp/qwen_audio_agent_online')
  if (!configuredState || repairState) {
    environment[STATE_KEY] = resolve(dataDirectory, 'memory/agent-memory')
    automaticKeys.push(STATE_KEY)
    if (repairState) repairedKeys.push(STATE_KEY)
  } else {
    // Keep existing databases and intentional custom directories in place.
    environment[STATE_KEY] = configuredState
  }
  return { environment, automaticKeys, repairedKeys }
}

// Persist selectors, never a build-machine/source-checkout absolute path.
// Re-read under the shared settings lock so concurrent settings saves survive.
export function persistAutomaticAgentMemoryPaths({
  configPath, rawEnvironment, automaticKeys, inheritedKeys = new Set(),
} = {}) {
  const candidates = automaticKeys.filter(key => !inheritedKeys.has(key))
  if (!candidates.length || !existsSync(configPath)) return []
  return withFileTransaction(configPath, () => {
    const original = readFileSync(configPath, 'utf8')
    const values = parseEnv(original)
    const changes = candidates.filter(key => clean(values[key]) !== 'auto'
      && clean(values[key]) === clean(rawEnvironment[key]))
    if (!changes.length) return []
    const pending = new Set(changes)
    const lines = original.split(/\r?\n/).map(line => {
      const key = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/)?.[1]
      if (!changes.includes(key)) return line
      pending.delete(key)
      return `${key}=auto`
    })
    while (lines.at(-1) === '') lines.pop()
    for (const key of pending) lines.push(`${key}=auto`)
    const content = lines.join('\n') + '\n'
    const backup = `${configPath}.before-agent-memory-auto`
    try {
      copyFileSync(configPath, backup, constants.COPYFILE_EXCL)
      chmodSync(backup, 0o600)
    } catch (error) {
      if (error.code !== 'EEXIST') throw error
    }
    const temporary = `${configPath}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, content, { encoding: 'utf8', mode: 0o600, flag: 'wx' })
      replaceFileSync(temporary, configPath)
      chmodSync(configPath, 0o600)
    } finally {
      try { unlinkSync(temporary) } catch { /* Preserve the original write error. */ }
    }
    return changes
  })
}

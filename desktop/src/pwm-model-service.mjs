import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { parse } from 'yaml'
import {
  PWM_FIELDS, PWM_SECRET_FIELDS, normalizePwmSettings, pwmSettingsPatch,
  pwmCompletionUrl, pwmEmbeddingUrl,
} from '../../shared/pwm-model-settings.mjs'

export function loadPwmYamlSettings({ env = process.env, sourceRoot } = {}) {
  const path = resolve(env.AGENT_MEMORY_CONFIG || resolve(sourceRoot, 'memory/config.yaml'))
  if (!existsSync(path)) return {}
  let pwm
  try { pwm = parse(readFileSync(path, 'utf8'))?.personal_world_model || {} }
  catch { throw new Error('PWM 配置文件解析失败，请检查 config.yaml 格式') }
  const expand = value => typeof value === 'string'
    ? value.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_, key) => env[key] || '')
    : value
  const values = {}
  for (const [field, [, section, key]] of Object.entries(PWM_FIELDS)) {
    if (pwm[section]?.[key] !== undefined) values[field] = expand(pwm[section][key])
  }
  if (!values.pwmEmbeddingApiKey && pwm.embedding_model?.api_key_env) {
    values.pwmEmbeddingApiKey = env[pwm.embedding_model.api_key_env] || ''
  }
  return values
}

// Probes carry only fixed sample text, never user memories. No configuration
// is written, redirects are rejected, and provider bodies/credentials never
// appear in IPC errors.
export async function testPwmModel(kind, draft, current, { fetchImpl = fetch } = {}) {
  if (!['llm', 'embedding'].includes(kind)) throw new Error('无效的 PWM 客户端')
  const prefix = kind === 'llm' ? 'pwmLlm' : 'pwmEmbedding'
  const patch = Object.fromEntries(Object.entries(pwmSettingsPatch(draft || {}))
    .filter(([field]) => field.startsWith(prefix)))
  const selected = Object.fromEntries(Object.entries(current).filter(([field]) => field.startsWith(prefix)))
  const settings = normalizePwmSettings({ ...selected, ...patch })
  const key = settings[`${prefix}ApiKey`]
  if (!key) return { ok: false, code: 'key' }
  // Match the runtime clients, which require an explicit API key.
  const headers = { 'Content-Type': 'application/json', ...(key ? { Authorization: `Bearer ${key}` } : {}) }
  const llmPayload = {
    model: settings.pwmLlmModel, messages: [{ role: 'user', content: 'Return a JSON object with ok set to true.' }],
    temperature: 0.2, stream: false, max_tokens: Math.min(settings.pwmLlmMaxTokens, 128),
    ...(settings.pwmLlmJsonMode ? { response_format: { type: 'json_object' } } : {}),
    thinking: { type: settings.pwmLlmThinking },
  }
  const without = (value, field) => Object.fromEntries(Object.entries(value).filter(([key]) => key !== field))
  const payloads = kind === 'llm'
    ? [llmPayload, without(llmPayload, 'thinking'), without(llmPayload, 'response_format')]
    : [{ model: settings.pwmEmbeddingModel, input: ['PWM connection test'] }]
  const signal = AbortSignal.timeout(Math.min(settings[`${prefix}Timeout`], 60) * 1000)
  const url = kind === 'llm' ? pwmCompletionUrl(settings.pwmLlmBaseUrl) : pwmEmbeddingUrl(settings.pwmEmbeddingBaseUrl)
  try {
    for (let index = 0; index < payloads.length; index++) {
      const response = await fetchImpl(url, { method: 'POST', headers, redirect: 'error', signal, body: JSON.stringify(payloads[index]) })
      if (!response.ok) {
        if ([400, 422].includes(response.status) && index < payloads.length - 1) { await response.body?.cancel(); continue }
        await response.body?.cancel()
        throw new Error(`PWM_HTTP_${response.status}`)
      }
      const data = await response.json()
      if (kind === 'llm') {
        if (!data?.choices?.[0]?.message?.content) throw new Error('PWM_INVALID_RESPONSE')
      } else {
        const vector = data?.data?.[0]?.embedding
        if (!Array.isArray(vector) || !vector.every(Number.isFinite)) throw new Error('PWM_INVALID_RESPONSE')
        if (vector.length !== settings.pwmEmbeddingDimensions) {
          return { ok: false, code: 'dimensions', dimensions: vector.length }
        }
      }
      return { ok: true }
    }
  } catch (error) {
    if (/^PWM_HTTP_\d+$/.test(error.message)) return { ok: false, code: 'http', status: Number(error.message.slice(9)) }
    if (signal.aborted) return { ok: false, code: 'timeout' }
    return { ok: false, code: error.message === 'PWM_INVALID_RESPONSE' ? 'response' : 'connection' }
  }
}

export function pwmSettingsChanged(previous, next) {
  return Object.keys(PWM_FIELDS).some(field => previous[field] !== next[field])
}

export function retainPwmSecrets(draft) {
  for (const field of PWM_SECRET_FIELDS) {
    if (!String(draft[field] || '').trim()) delete draft[field]
  }
  return draft
}

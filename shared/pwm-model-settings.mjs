// Shared schema for desktop settings and the Python PWM environment overrides.
// No Node dependencies: the settings renderer also consumes this contract.
export const PWM_FIELDS = {
  pwmLlmBaseUrl: ['LLM_BASE_URL', 'llm_model', 'llm_base_url', 'https://api.deepseek.com'],
  pwmLlmModel: ['LLM_MODEL', 'llm_model', 'llm_name', 'deepseek-v4-flash'],
  pwmLlmApiKey: ['LLM_API_KEY', 'llm_model', 'llm_api_key', ''],
  pwmLlmTimeout: ['LLM_TIMEOUT', 'llm_model', 'llm_timeout', 120, 1, 600],
  pwmLlmMaxTokens: ['LLM_MAX_TOKENS', 'llm_model', 'llm_max_tokens', 8192, 1, 262144],
  pwmLlmThinking: ['LLM_THINKING', 'llm_model', 'llm_thinking', 'disabled'],
  pwmLlmJsonMode: ['LLM_JSON_MODE', 'llm_model', 'llm_json_mode', true],
  pwmEmbeddingBaseUrl: ['EMBEDDING_BASE_URL', 'embedding_model', 'base_url', 'https://api.z.ai/api/paas/v4'],
  pwmEmbeddingModel: ['EMBEDDING_MODEL', 'embedding_model', 'model', 'embedding-3'],
  pwmEmbeddingApiKey: ['EMBEDDING_API_KEY', 'embedding_model', 'api_key', ''],
  pwmEmbeddingDimensions: ['EMBEDDING_DIMENSIONS', 'embedding_model', 'dimensions', 2048, 8, 65536],
  pwmEmbeddingBatchSize: ['EMBEDDING_BATCH_SIZE', 'embedding_model', 'batch_size', 32, 1, 1024],
  pwmEmbeddingTimeout: ['EMBEDDING_TIMEOUT', 'embedding_model', 'timeout', 30, 1, 600],
  pwmEmbeddingNormalize: ['EMBEDDING_NORMALIZE', 'embedding_model', 'normalize', true],
}
export const PWM_SETTING_KEYS = Object.fromEntries(Object.entries(PWM_FIELDS)
  .map(([field, [key]]) => [field, `QWEN_AUDIO_PWM_${key}`]))
export const PWM_DEFAULTS = Object.fromEntries(Object.entries(PWM_FIELDS)
  .map(([field, spec]) => [field, spec[3]]))
export const PWM_SECRET_FIELDS = ['pwmLlmApiKey', 'pwmEmbeddingApiKey']

export function pwmSettingsPatch(settings) {
  return Object.fromEntries(Object.entries(settings).filter(([key, value]) => (
    !PWM_SECRET_FIELDS.includes(key) || String(value || '').trim()
  )))
}

export function parsePwmSettings(environment = {}, defaults = {}) {
  return Object.fromEntries(Object.entries(PWM_FIELDS).map(([field, spec]) => {
    const value = environment[PWM_SETTING_KEYS[field]] ?? defaults[field] ?? spec[3]
    return [field, typeof spec[3] === 'boolean'
      ? ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase())
      : typeof spec[3] === 'number' ? Number(value) : String(value).trim()]
  }))
}

export function pwmHttpUrl(value) {
  const url = new URL(String(value).trim())
  if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password
      || url.search || url.hash) {
    throw new Error('PWM 服务地址须为 HTTP 或 HTTPS，不能包含凭据、查询参数或片段')
  }
  return url.href.replace(/\/+$/, '')
}

export function normalizePwmSettings(settings = {}) {
  const result = { ...PWM_DEFAULTS }
  for (const [field, spec] of Object.entries(PWM_FIELDS)) {
    const value = settings[field] ?? spec[3]
    if (typeof spec[3] === 'number') {
      const number = Number(value)
      if (!Number.isInteger(number) || number < spec[4] || number > spec[5]) {
        throw new Error(`PWM 参数 ${spec[0]} 必须是 ${spec[4]}–${spec[5]} 的整数`)
      }
      result[field] = number
    } else if (typeof spec[3] === 'boolean') {
      result[field] = typeof value === 'boolean' ? value : ['1', 'true'].includes(String(value))
    } else {
      result[field] = String(value).trim()
      if (/[\r\n\0]/.test(result[field])) throw new Error('PWM 参数不能包含换行或空字符')
    }
  }
  for (const field of ['pwmLlmBaseUrl', 'pwmEmbeddingBaseUrl']) result[field] = pwmHttpUrl(result[field])
  for (const field of ['pwmLlmModel', 'pwmEmbeddingModel']) {
    if (!result[field]) throw new Error('请填写 PWM 模型名称')
  }
  if (!['disabled', 'enabled'].includes(result.pwmLlmThinking)) throw new Error('无效的 PWM Thinking 参数')
  return result
}

export function publicPwmSettings(settings) {
  const result = { ...settings }
  for (const field of PWM_SECRET_FIELDS) {
    result[`${field}Configured`] = Boolean(settings[field])
    result[field] = ''
  }
  return result
}

export function pwmCompletionUrl(baseUrl) {
  const base = pwmHttpUrl(baseUrl)
  return `${base === 'https://api.deepseek.com' ? `${base}/v1` : base}/chat/completions`
}

export function pwmEmbeddingUrl(baseUrl) {
  const base = pwmHttpUrl(baseUrl)
  if (base.endsWith('/embeddings')) return base
  return /\/v1$|\/api\/paas\/v\d+$/.test(base) ? `${base}/embeddings` : `${base}/v1/embeddings`
}

import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createServer } from 'node:http'
import test from 'node:test'
import { PWM_DEFAULTS, PWM_SETTING_KEYS, normalizePwmSettings, parsePwmSettings, publicPwmSettings } from '../../shared/pwm-model-settings.mjs'
import { parseSettings, updateSettingsContent, applySettingsEnvironment, clientSettingsPatch } from '../src/settings-config.mjs'
import { loadPwmYamlSettings, testPwmModel, pwmSettingsChanged } from '../src/pwm-model-service.mjs'
import { createSettingsStore } from '../src/settings-store.mjs'

function temporary(t) {
  const root = mkdtempSync(join(tmpdir(), 'pwm-settings-'))
  t.after(() => rmSync(root, { recursive: true, force: true }))
  return root
}

test('PWM defaults follow YAML, config.env overrides YAML, and models have independent keys', t => {
  const root = temporary(t)
  const yaml = join(root, 'config.yaml')
  writeFileSync(yaml, 'personal_world_model:\n  llm_model:\n    llm_name: yaml-chat\n    llm_api_key: ${CHAT_KEY}\n  embedding_model:\n    api_key_env: VECTOR_KEY\n    dimensions: 16\n')
  const env = { AGENT_MEMORY_CONFIG: yaml, CHAT_KEY: 'llm-secret', VECTOR_KEY: 'embedding-secret' }
  const store = createSettingsStore({ configDir: join(root, 'gateway'), clientDir: join(root, 'client'), env,
    pwmDefaults: env => loadPwmYamlSettings({ env, sourceRoot: root }) })
  const initial = store.load()
  assert.equal(initial.pwmLlmModel, 'yaml-chat')
  assert.equal(initial.pwmLlmApiKey, 'llm-secret')
  assert.equal(initial.pwmEmbeddingApiKey, 'embedding-secret')
  assert.equal(initial.pwmEmbeddingDimensions, 16)
  store.save({ language: 'en', pwmLlmModel: 'new-chat', pwmLlmApiKey: '', pwmEmbeddingApiKey: 'new-vector-key', pwmLlmJsonMode: false })
  assert.equal(store.load().pwmLlmApiKey, 'llm-secret')
  assert.equal(store.load().pwmEmbeddingApiKey, 'new-vector-key')
  assert.equal(store.load().pwmLlmModel, 'new-chat')
  assert.equal(env.QWEN_AUDIO_PWM_LLM_JSON_MODE, 'false')
  assert.doesNotMatch(readFileSync(store.clientSettingsPath, 'utf8'), /PWM|secret/)
  assert.doesNotMatch(readFileSync(store.path, 'utf8'), /llm-secret/)
  const publicValue = publicPwmSettings(store.load())
  assert.equal(publicValue.pwmLlmApiKey, '')
  assert.equal(publicValue.pwmLlmApiKeyConfigured, true)
  assert.doesNotMatch(JSON.stringify(publicValue), /llm-secret|new-vector-key/)
})

test('PWM persistence validates numeric ranges and keeps blank keys and unknown config intact', () => {
  const current = 'CUSTOM=keep\nQWEN_AUDIO_PWM_LLM_API_KEY=old-key\n'
  const content = updateSettingsContent(current, { pwmLlmApiKey: '', pwmLlmTimeout: 60, pwmEmbeddingNormalize: false })
  assert.equal(parseSettings(content).pwmLlmApiKey, 'old-key')
  assert.match(content, /CUSTOM=keep/)
  assert.equal(parseSettings(content).pwmEmbeddingNormalize, false)
  assert.throws(() => updateSettingsContent(current, { pwmEmbeddingDimensions: 0 }))
  assert.throws(() => normalizePwmSettings({ pwmLlmBaseUrl: 'https://user:secret@example.org/v1' }))
  assert.throws(() => updateSettingsContent(current, { pwmLlmModel: 'a\nINJECT=value' }))
  assert.equal(updateSettingsContent(current, { pwmLlmModel: 'other' }, { scope: 'client' }), '\n')
  assert.deepEqual(clientSettingsPatch({ pwmLlmApiKey: 's', language: 'en' }), { language: 'en' })
  const env = { QWEN_AUDIO_PWM_LLM_API_KEY: 'existing' }
  applySettingsEnvironment({ pwmLlmApiKey: '', pwmEmbeddingModel: 'new-vector' }, env)
  assert.equal(env.QWEN_AUDIO_PWM_LLM_API_KEY, 'existing')
  assert.equal(env.QWEN_AUDIO_PWM_EMBEDDING_MODEL, 'new-vector')
  assert.equal(pwmSettingsChanged(PWM_DEFAULTS, { ...PWM_DEFAULTS, pwmEmbeddingApiKey: 's' }), true)
})

test('all desktop PWM environment keys map to round-trippable typed settings', () => {
  const values = { ...PWM_DEFAULTS, pwmLlmJsonMode: false, pwmEmbeddingNormalize: false, pwmEmbeddingTimeout: 42 }
  const environment = Object.fromEntries(Object.entries(PWM_SETTING_KEYS).map(([field, key]) => [key, String(values[field])]))
  assert.deepEqual(parsePwmSettings(environment), values)
})

test('connection probes send draft model/key via main-process HTTP and enforce returned dimensions', async t => {
  const requests = []
  const server = createServer(async (request, response) => {
    let body = ''
    for await (const data of request) body += data
    requests.push({ url: request.url, key: request.headers.authorization, payload: JSON.parse(body) })
    response.setHeader('Content-Type', 'application/json')
    response.end(JSON.stringify(request.url.endsWith('/embeddings')
      ? { data: [{ index: 0, embedding: Array(8).fill(0.1) }] }
      : { choices: [{ message: { content: '{"ok":true}' } }] }))
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  t.after(() => new Promise(resolve => server.close(resolve)))
  const url = `http://127.0.0.1:${server.address().port}/v1`
  const current = { ...PWM_DEFAULTS, pwmLlmApiKey: 'chat-key', pwmEmbeddingApiKey: 'vector-key' }
  assert.deepEqual(await testPwmModel('llm', { pwmLlmBaseUrl: url, pwmLlmModel: 'draft-chat', pwmLlmApiKey: '' }, current), { ok: true })
  assert.equal(requests[0].url, '/v1/chat/completions')
  assert.equal(requests[0].key, 'Bearer chat-key')
  assert.equal(requests[0].payload.model, 'draft-chat')
  const result = await testPwmModel('embedding', { pwmEmbeddingBaseUrl: url, pwmEmbeddingDimensions: 16 }, current)
  assert.deepEqual(result, { ok: false, code: 'dimensions', dimensions: 8 })
  assert.deepEqual(await testPwmModel('embedding', { pwmEmbeddingBaseUrl: url, pwmEmbeddingDimensions: 8 }, current), { ok: true })
  assert.equal(requests[1].key, 'Bearer vector-key')
  assert.deepEqual(await testPwmModel('llm', {}, PWM_DEFAULTS), { ok: false, code: 'key' })
})

test('probe supports LLM parameter fallback and never returns provider body or secret errors', async () => {
  const current = { ...PWM_DEFAULTS, pwmLlmApiKey: 'do-not-return' }
  let count = 0
  const success = await testPwmModel('llm', {}, current, { fetchImpl: async (_url, options) => {
    count++
    if (count === 1) return new Response('unsupported thinking', { status: 400 })
    assert.equal(JSON.parse(options.body).thinking, undefined)
    return Response.json({ choices: [{ message: { content: '{}' } }] })
  } })
  assert.deepEqual(success, { ok: true })
  const fail = await testPwmModel('llm', {}, current, { fetchImpl: async () => new Response('do-not-return', { status: 401 }) })
  assert.deepEqual(fail, { ok: false, code: 'http', status: 401 })
  const network = await testPwmModel('llm', {}, current, { fetchImpl: async () => { throw new Error('do-not-return') } })
  assert.deepEqual(network, { ok: false, code: 'connection' })
})

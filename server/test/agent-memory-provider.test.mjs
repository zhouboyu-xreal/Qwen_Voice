import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  normalizeMemoryProviderSelection,
} from '../../shared/memory-provider-catalog.mjs'
import {
  createConfiguredMemoryProvider,
} from '../src/app/memory-provider-factory.mjs'
import {
  AgentMemoryProvider,
} from '../src/conversation/memory/providers/agent-memory/provider.mjs'

test('selects Agent Memory through configuration', async () => {
  assert.equal(normalizeMemoryProviderSelection('Agent-Memory'), 'agent-memory')
  const stateDirectory = mkdtempSync(join(tmpdir(), 'qwaudio-agent-memory-built-in-'))
  const sidecarPath = join(stateDirectory, 'sidecar.py')
  writeFileSync(sidecarPath, '')
  const provider = createConfiguredMemoryProvider({
    config: {
      memoryProvider: 'agent-memory',
      agentMemoryStateDirectory: stateDirectory,
      agentMemoryPython: '',
      agentMemorySidecarPath: sidecarPath,
      agentMemoryConfigPath: '',
    },
    logger: { warn() {} },
    env: {},
  })
  assert.equal(provider.describe().key, 'agent-memory')
  assert.equal(provider.describe().capabilities.sessionObservation, true)
  await provider.close()
})

test('uses the bundled Agent Memory sidecar without external path settings', async () => {
  const provider = new AgentMemoryProvider({
    stateDirectory: mkdtempSync(join(tmpdir(), 'qwaudio-agent-memory-bundled-')),
    env: {},
  })
  assert.equal(provider.describe().key, 'agent-memory')
  await provider.close()
})

test('forwards completed sessions and semantic recall to the sidecar', async () => {
  const calls = []
  const sidecar = {
    lastError: null,
    request(method, params, options) {
      calls.push({ method, params, options })
      return Promise.resolve(method === 'recall'
        ? { status: 'ok', memory_context: '用户喜欢喝茶。' }
        : { observed: true })
    },
    close() {},
  }
  const provider = new AgentMemoryProvider({
    stateDirectory: mkdtempSync(join(tmpdir(), 'qwaudio-agent-memory-')),
    sidecar,
  })

  const result = await provider.query('owner-a', '我喜欢什么？', {}, {
    promptLanguage: 'zh',
  })
  assert.equal(result.context, '用户喜欢喝茶。')
  assert.deepEqual(result.memories, [])
  assert.notEqual(calls[0].params.ownerId, 'owner-a')
  assert.equal(calls[0].params.promptLanguage, 'zh')

  await provider.observe('owner-a', {
    messages: [
      {
        id: 'u1',
        role: 'user',
        content: '我喜欢喝茶。',
        turnId: 'turn-1',
      },
      {
        id: 'a1',
        role: 'assistant',
        content: '记住了。',
        turnId: 'turn-1',
      },
    ],
  }, { sessionId: 'session-1' })
  await provider.flush('owner-a', { sessionId: 'session-1' })
  await provider.finalizeSession('owner-a', { sessionId: 'session-1' })

  const observe = calls.find(call => call.method === 'observe')
  assert.equal(observe.params.messages.length, 2)
  assert.equal(observe.params.sessionId, 'session-1')
  const checkpoint = calls.find(call => call.method === 'finalize' && call.params.boundary === 'checkpoint')
  assert.equal(checkpoint.options.timeoutMs, 120_000)
  const finalize = calls.find(call => call.method === 'finalize' && call.params.boundary === 'session_end')
  assert.equal(finalize.params.sessionId, 'session-1')
  assert.equal(finalize.options.timeoutMs, 120_000)
})

test('does not persist or expose a local document snapshot', () => {
  const provider = new AgentMemoryProvider({
    stateDirectory: mkdtempSync(join(tmpdir(), 'qwaudio-agent-memory-documents-')),
    sidecar: { lastError: null, close() {} },
  })
  assert.deepEqual(provider.list('owner-a'), [])
  assert.throws(
    () => provider.apply('owner-a', [{ document: 'user', append: '- 船长' }]),
    /不支持本地 memory 文档编辑/,
  )
})

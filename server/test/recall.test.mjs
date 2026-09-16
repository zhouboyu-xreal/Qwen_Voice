import assert from 'node:assert/strict'
import test from 'node:test'
import { TaskManager } from '../src/task/task-manager.mjs'
import { ToolCallHandler } from '../src/voice/tools/tool-call-handler.mjs'
import {
  FRONTEND_MEMORY_RECALL_CAPABILITY,
  MEMORY_RECALL_TOOL_NAME,
  frontendTools,
} from '../src/voice/frontend-tools.mjs'
import { TurnTranscripts } from '../src/voice/tools/turn-transcripts.mjs'

function harness({ memoryService = null } = {}) {
  const outputs = []
  const handler = new ToolCallHandler({
    taskManager: new TaskManager(),
    ownerId: 'owner',
    sessionId: 'voice',
    transcripts: new TurnTranscripts({ waitMs: 5 }),
    getFrontend: () => ({
      sendFunctionOutput: async (...args) => outputs.push(args),
      ensureResponse: async () => {},
    }),
    getTurnId: () => 'turn-one',
    getTurnGeneration: () => 1,
    backendRuntime: { run: async () => ({ content: '完成', metadata: {} }) },
    memoryService,
  })
  return {
    handler,
    lastOutput: () => outputs.at(-1)[1],
  }
}

const call = args => ({
  name: MEMORY_RECALL_TOOL_NAME,
  call_id: 'call-1',
  arguments: JSON.stringify(args),
})

test('memory_recall queries the configured semantic memory provider', async () => {
  const calls = []
  const { handler, lastOutput } = harness({
    memoryService: {
      query: async (...args) => {
        calls.push(args)
        return { context: '用户下周去北京开会。' }
      },
    },
  })

  await handler.handle(call({ query: '我下周有什么安排？', limit: 3 }))

  assert.deepEqual(calls, [[
    'owner',
    '我下周有什么安排？',
    { limit: 3 },
    {
      source: 'realtime-memory-recall-tool',
      sessionId: 'voice',
      turnId: 'turn-one',
      traceId: 'call-1',
    },
  ]])
  assert.deepEqual(lastOutput(), {
    status: 'found',
    context: '用户下周去北京开会。',
  })
})

test('memory_recall reports a miss without inventing a memory', async () => {
  const { handler, lastOutput } = harness({
    memoryService: { query: async () => ({ context: '' }) },
  })

  await handler.handle(call({ query: '我养的猫叫什么？' }))

  assert.equal(lastOutput().status, 'not_found')
  assert.match(lastOutput().message, /可靠长期记忆/)
})

test('memory_recall reports provider failures as retryable', async () => {
  const { handler, lastOutput } = harness({
    memoryService: { query: async () => { throw new Error('sidecar unavailable') } },
  })

  await handler.handle(call({ query: '我的偏好是什么？' }))

  assert.equal(lastOutput().status, 'failed')
  assert.equal(lastOutput().error_code, 'memory_recall_failed')
  assert.equal(lastOutput().retryable, true)
})

test('memory_recall is exposed only for a semantic memory provider', () => {
  const names = context => frontendTools(context).map(tool => tool.function.name)
  assert.ok(!names({}).includes(MEMORY_RECALL_TOOL_NAME))
  assert.ok(names({
    frontend: { capabilities: [FRONTEND_MEMORY_RECALL_CAPABILITY] },
  }).includes(MEMORY_RECALL_TOOL_NAME))
})

test('memory_recall requires a focused query and describes its data boundary', () => {
  const [tool] = frontendTools({
    frontend: { capabilities: [FRONTEND_MEMORY_RECALL_CAPABILITY] },
  }).filter(item => item.function.name === MEMORY_RECALL_TOOL_NAME)

  assert.deepEqual(tool.function.parameters.required, ['query'])
  assert.match(tool.function.description, /长期记忆/)
  assert.match(tool.function.description, /不用于公开网页、知识库文档/)
  assert.match(tool.function.description, /不能补全或编造/)
})

test('memory_recall is unavailable when no semantic memory provider exists', async () => {
  const { handler, lastOutput } = harness()

  await handler.handle(call({ query: '以前聊过什么？' }))

  assert.equal(lastOutput().status, 'failed')
  assert.equal(lastOutput().error_code, 'tool_unavailable')
})

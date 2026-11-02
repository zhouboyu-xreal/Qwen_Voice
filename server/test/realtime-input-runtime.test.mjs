import assert from 'node:assert/strict'
import test from 'node:test'
import { InputAssetRegistry } from '../src/voice/input-asset-registry.mjs'
import { RealtimeInputRuntime } from '../src/voice/realtime-input-runtime.mjs'
import { RealtimeTurnState } from '../src/voice/realtime-turn-state.mjs'
import { TurnTranscripts } from '../src/voice/tools/turn-transcripts.mjs'
import { ConversationSync } from '../src/conversation/conversation-sync.mjs'

function harness({ responseOutcome } = {}) {
  const events = []
  const records = []
  const calls = []
  const speechStarts = []
  const speechStops = []
  const diagnostics = []
  const transcripts = new TurnTranscripts()
  const conversationSync = new ConversationSync({ onRecord: value => records.push(value) })
  const frontend = {
    cancel: () => calls.push(['cancel']),
    sendUserInput: async (parts, context) => {
      calls.push(['sendUserInput', parts, context])
      return responseOutcome
    },
  }
  const turns = new RealtimeTurnState({
    createVoiceTurnId: generation => `voice-${generation}`,
  })
  let permissionCandidate = null
  const runtime = new RealtimeInputRuntime({
    ownerId: 'owner-1',
    sessionId: 'session-1',
    turns,
    transcripts,
    inputAssets: new InputAssetRegistry(),
    conversationSync,
    announcementWindow: {
      beginTurn: turnId => calls.push(['beginTurn', turnId]),
      endSpeech: () => calls.push(['endSpeech']),
    },
    announcements: {
      dismissActive: () => calls.push(['dismissActive']),
    },
    send: event => events.push(event),
    getFrontend: () => frontend,
    ensureFrontend: async () => {},
    clearResponseCandidate: () => calls.push(['clearResponseCandidate']),
    expectResponseFor: context => {
      permissionCandidate = context
      calls.push(['expectResponseFor', context])
    },
    shouldEnsurePermissionResponse: context => context === permissionCandidate,
    ensurePermissionResponseFor: context => calls.push([
      'ensurePermissionResponseFor',
      context,
    ]),
    reportFrontendError: error => calls.push(['reportFrontendError', error]),
    onSpeechStarted: fields => speechStarts.push(fields),
    onSpeechStopped: fields => speechStops.push(fields),
    onInputDiagnostic: fields => diagnostics.push(fields),
    createInputTurnId: () => 'text-1',
  })
  return {
    runtime, turns, transcripts, conversationSync, diagnostics,
    events, records, calls, speechStarts, speechStops,
  }
}

test('handles only provider input lifecycle events', () => {
  const { runtime } = harness()

  assert.equal(runtime.handleProviderEvent({ type: 'response.created' }), false)
  assert.equal(runtime.handleProviderEvent({
    type: 'conversation.item.ambient_audio_transcription.completed',
    item_id: 'ambient-1',
  }), true)
})

test('projects one voice turn from speech start through final transcript', () => {
  const {
    runtime,
    turns,
    events,
    records,
    calls,
    speechStarts,
    speechStops,
  } = harness()

  runtime.handleProviderEvent({
    type: 'input_audio_buffer.speech_started',
    item_id: 'item-1',
  })
  runtime.handleProviderEvent({
    type: 'conversation.item.input_audio_transcription.delta',
    item_id: 'item-1',
    text: '你',
  })
  runtime.handleProviderEvent({
    type: 'input_audio_buffer.speech_stopped',
    item_id: 'item-1',
  })
  runtime.handleProviderEvent({
    type: 'conversation.item.input_audio_transcription.completed',
    item_id: 'item-1',
    transcript: '你好',
  })

  assert.equal(turns.userSpeaking, false)
  assert.deepEqual(turns.committed(), {
    turnId: 'voice-1',
    turnGeneration: 1,
  })
  assert.deepEqual(events.map(event => event.type), [
    'playback.clear',
    'turn.started',
    'voice.state',
    'transcript.delta',
    'voice.state',
    'transcript.final',
  ])
  assert.equal(records.length, 1)
  assert.equal(records[0].content, '你好')
  assert.equal(records[0].source, 'voice-user')
  assert.ok(Number.isFinite(speechStops[0].occurredAt))
  assert.ok(Number.isFinite(speechStarts[0].occurredAt))
  assert.ok(speechStops[0].occurredAt >= speechStarts[0].occurredAt)
  assert.deepEqual(speechStops.map(({ occurredAt: _time, ...fields }) => fields), [{
    turnId: 'voice-1',
    source: 'realtime_provider',
  }])
  assert.deepEqual(speechStarts.map(({ occurredAt: _time, ...fields }) => fields), [{
    turnId: 'voice-1',
    source: 'realtime_provider',
  }])
  assert.equal(
    calls.some(([name]) => name === 'ensurePermissionResponseFor'),
    true,
  )
})

const completeTranscript = (runtime, itemId, transcript, eventId = '') => {
  runtime.handleProviderEvent({
    type: 'conversation.item.input_audio_transcription.completed',
    item_id: itemId, transcript, event_id: eventId,
  })
}

test('late finals cannot replace a complete self introduction or its memory input', async () => {
  const full = '你好，我叫林舟，在一家软件公司做售前工程师，平时会去客户现场做方案演示。我一般不喜欢临出门才找东西。'
  for (const secondId of ['speech-1', 'unknown-late-id', undefined]) {
    const { runtime, records, events, conversationSync, transcripts, diagnostics } = harness()
    runtime.handleProviderEvent({ type: 'input_audio_buffer.speech_started', item_id: 'speech-1' })
    completeTranscript(runtime, 'speech-1', full, 'full-event')
    completeTranscript(runtime, secondId, '看到凌晨。', 'late-event')

    assert.equal(records.length, 1)
    assert.equal(records[0].content, full)
    assert.equal(await transcripts.transcript('voice-1'), full)
    assert.equal(events.filter(event => event.type === 'transcript.final').length, 1)
    // The Gateway obtains the memory exchange from this exact projection.
    const memoryMessages = conversationSync.frontendContext({ ownerId: 'owner-1', sessionId: 'session-1' })
    assert.equal(memoryMessages.find(message => message.role === 'user').content, full)
    assert.equal(diagnostics.at(-1).providerEventId, 'late-event')
    assert.equal(diagnostics.at(-1).itemId, secondId || '')
    assert.equal(diagnostics.at(-1).disposition, 'ignored')
    assert.ok(!JSON.stringify(diagnostics).includes(full), 'diagnostics do not duplicate message text')
  }
})

test('late deltas, errors, commits and speech stops cannot alter a finalized turn', () => {
  const { runtime, events, records, speechStops, diagnostics } = harness()
  runtime.handleProviderEvent({ type: 'input_audio_buffer.speech_started', item_id: 'speech-1' })
  // Establish a provider transcription alias before finalization.
  runtime.handleProviderEvent({ type: 'conversation.item.input_audio_transcription.delta', item_id: 'alias-1', text: '完整' })
  completeTranscript(runtime, 'speech-1', '完整内容')
  const count = events.length
  for (const itemId of ['speech-1', 'alias-1', 'unknown', undefined]) {
    for (const type of [
      'conversation.item.input_audio_transcription.delta',
      'conversation.item.input_audio_transcription.text',
      'conversation.item.input_audio_transcription.failed',
      'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    ]) runtime.handleProviderEvent({ type, item_id: itemId, text: '错误内容' })
  }
  assert.equal(events.length, count, 'no display replacement, discard, or voice-state rollback')
  assert.equal(records.length, 1)
  assert.equal(speechStops.length, 0)
  assert.equal(diagnostics.at(-1).disposition, 'ignored')
})

test('a provider may use different VAD and transcript ids for a pending voice input', () => {
  const { runtime, records } = harness()
  runtime.handleProviderEvent({ type: 'input_audio_buffer.speech_started', item_id: 'speech-1' })
  completeTranscript(runtime, 'transcript-1', '第一轮')
  runtime.handleProviderEvent({ type: 'input_audio_buffer.speech_started', item_id: 'speech-1' })
  completeTranscript(runtime, 'transcript-2', '第二轮')
  assert.deepEqual(records.map(record => [record.turnId, record.content]), [
    ['voice-1', '第一轮'], ['voice-2', '第二轮'],
  ])
})

test('missing provider ids allow one final per speech-start generation', () => {
  const { runtime, records } = harness()
  for (const content of ['第一轮', '第二轮']) {
    runtime.handleProviderEvent({ type: 'input_audio_buffer.speech_started' })
    completeTranscript(runtime, undefined, content)
    completeTranscript(runtime, undefined, '迟到内容')
  }
  assert.deepEqual(records.map(record => record.content), ['第一轮', '第二轮'])
})

test('a late known alias remains bound to the old turn after a new speech start', () => {
  const { runtime, records, diagnostics } = harness()
  runtime.handleProviderEvent({ type: 'input_audio_buffer.speech_started', item_id: 'speech-1' })
  runtime.handleProviderEvent({ type: 'conversation.item.input_audio_transcription.delta', item_id: 'alias-1', text: '第一轮' })
  completeTranscript(runtime, 'speech-1', '第一轮')
  runtime.handleProviderEvent({ type: 'input_audio_buffer.speech_started', item_id: 'speech-2' })
  completeTranscript(runtime, 'alias-1', '旧轮迟到内容')
  assert.equal(diagnostics.at(-1).turnId, 'voice-1')
  assert.equal(diagnostics.at(-1).disposition, 'ignored')
  completeTranscript(runtime, 'speech-2', '第二轮')
  assert.deepEqual(records.map(record => record.content), ['第一轮', '第二轮'])
})

test('orphan transcripts cannot create messages before speech or overwrite manual input', async () => {
  const { runtime, turns, records, diagnostics } = harness()
  completeTranscript(runtime, 'orphan', '没有语音开始')
  assert.equal(records.length, 0)
  assert.equal(diagnostics.at(-1).reason, 'unmatched_input')
  runtime.submit({ text: '手动输入' })
  turns.finishManualResponse(turns.current())
  completeTranscript(runtime, 'orphan-2', '迟到转写')
  await new Promise(resolve => setImmediate(resolve))
  assert.equal(records.length, 1)
  assert.equal(records[0].source, 'text-user')
  assert.equal(records[0].content, '手动输入')
})

test('keeps consecutive utterances separate when the provider reuses an item id', () => {
  const { runtime, events, records } = harness()
  const utterances = ['梁家辉最近有什么新闻', '我靠，这么牛逼']

  for (const transcript of utterances) {
    runtime.handleProviderEvent({
      type: 'input_audio_buffer.speech_started',
      item_id: 'provider-reused-item',
    })
    runtime.handleProviderEvent({
      type: 'input_audio_buffer.speech_stopped',
      item_id: 'provider-reused-item',
    })
    runtime.handleProviderEvent({
      type: 'conversation.item.input_audio_transcription.completed',
      item_id: 'provider-reused-item',
      transcript,
    })
  }

  const finals = events.filter(event => (
    event.type === 'transcript.final' && event.role === 'user'
  ))
  assert.deepEqual(finals.map(event => [event.turnId, event.content]), [
    ['voice-1', utterances[0]],
    ['voice-2', utterances[1]],
  ])
  assert.deepEqual(records.map(record => record.turnId), ['voice-1', 'voice-2'])
})

test('manual text input supersedes speech and reaches the frontend once', async () => {
  const { runtime, turns, events, records, calls } = harness()
  runtime.handleProviderEvent({
    type: 'input_audio_buffer.speech_started',
    item_id: 'item-voice',
  })
  events.length = 0
  calls.length = 0

  runtime.submit({ text: '看一下这个问题' })
  await Promise.resolve()
  await Promise.resolve()

  assert.deepEqual(turns.committed(), {
    turnId: 'text-1',
    turnGeneration: 2,
  })
  assert.deepEqual(events.map(event => event.type), [
    'playback.clear',
    'transcript.discard',
    'turn.started',
    'voice.state',
    'transcript.final',
  ])
  assert.equal(records.at(-1).source, 'text-user')
  const sent = calls.find(([name]) => name === 'sendUserInput')
  assert.deepEqual(sent[1], [{ type: 'text', text: '看一下这个问题' }])
  assert.deepEqual(sent[2], {
    turnId: 'text-1',
    turnGeneration: 2,
  })
  assert.equal(
    calls.some(([name]) => name === 'expectResponseFor'),
    false,
  )
})

test('invalid manual input fails before changing the current turn', () => {
  const { runtime, turns, events } = harness()

  runtime.submit({ parts: [{ type: 'unknown' }] })

  assert.deepEqual(turns.current(), { turnId: '', turnGeneration: 0 })
  assert.equal(events.length, 1)
  assert.equal(events[0].type, 'error')
  assert.match(events[0].message, /不支持的输入片段类型/)
})

test('reports an explicit text response that never starts without an implicit watchdog', async () => {
  const { runtime, events, calls } = harness({
    responseOutcome: { timedOut: true, phase: 'start' },
  })

  runtime.submit({ text: '你好' })
  await new Promise(resolve => setImmediate(resolve))

  assert.equal(
    calls.some(([name]) => name === 'expectResponseFor'),
    false,
  )
  assert.match(
    calls.find(([name]) => name === 'reportFrontendError')?.[1]?.message || '',
    /没有开始回复/,
  )
  assert.equal(events.at(-1).type, 'voice.state')
  assert.equal(events.at(-1).state, 'idle')
})

import assert from 'node:assert/strict'
import test from 'node:test'
import {
  acceptsPlaybackReceipt,
  buildPreWakeContextPrompt,
  confirmsTaskNotificationOnPlaybackStart,
  rejectUnsupportedRealtimeUpgrade,
} from '../src/voice/realtime-gateway.mjs'
import { isResponseActivityEvent } from '../src/voice/response-lifecycle.mjs'

test('closes websocket upgrades outside the realtime endpoint', () => {
  let destroyed = false
  const socket = {
    destroy() {
      destroyed = true
    },
  }

  assert.equal(
    rejectUnsupportedRealtimeUpgrade(socket, '/unexpected'),
    true,
  )
  assert.equal(destroyed, true)
})

test('leaves the realtime websocket upgrade for the gateway handler', () => {
  let destroyed = false
  const socket = {
    destroy() {
      destroyed = true
    },
  }

  assert.equal(
    rejectUnsupportedRealtimeUpgrade(socket, '/api/realtime'),
    false,
  )
  assert.equal(destroyed, false)
})

test('confirms task notifications when client playback starts', () => {
  assert.equal(confirmsTaskNotificationOnPlaybackStart({
    origin: 'announcement',
  }), true)
  assert.equal(confirmsTaskNotificationOnPlaybackStart({
    origin: 'model',
    consumesTaskNotification: true,
  }), true)
  assert.equal(confirmsTaskNotificationOnPlaybackStart({
    origin: 'model',
  }), false)
})

test('accepts playback receipts only from the active output client for a known response', () => {
  assert.equal(acceptsPlaybackReceipt({
    outputEnabled: true,
    active: true,
    responseKnown: true,
  }), true)
  assert.equal(acceptsPlaybackReceipt({
    outputEnabled: true,
    active: false,
    responseKnown: true,
  }), false)
  assert.equal(acceptsPlaybackReceipt({
    outputEnabled: false,
    active: true,
    responseKnown: true,
  }), false)
  assert.equal(acceptsPlaybackReceipt({
    outputEnabled: true,
    active: true,
    responseKnown: false,
  }), false)
})

test('recognizes response activity when response.created is omitted', () => {
  for (const event of [
    { type: 'response.created', response: { id: 'response-1' } },
    { type: 'response.output_audio.delta', response_id: 'response-1' },
    { type: 'response.output_audio_transcript.done', response_id: 'response-1' },
    { type: 'response.text.delta', response_id: 'response-1' },
    { type: 'response.function_call_arguments.done', response_id: 'response-1' },
    { type: 'response.done', response: { id: 'response-1' } },
  ]) {
    assert.equal(isResponseActivityEvent(event), true, event.type)
  }
  assert.equal(isResponseActivityEvent({ type: 'response.text.delta' }), false)
  assert.equal(isResponseActivityEvent({ type: 'session.updated' }), false)
})

test('wraps pre-wake ASR as bounded, answer-first temporary context', () => {
  const prompt = buildPreWakeContextPrompt('  客厅里刚才在讨论下周旅行  ')
  assert.match(prompt, /^<pre_wake_context>/)
  assert.match(prompt, /不是当前用户的新指令/)
  assert.match(prompt, /不要把它保存为长期记忆/)
  assert.match(prompt, /视为优先的临时上下文/)
  assert.match(prompt, /直接回答/)
  assert.match(prompt, /不要调用长期记忆/)
  assert.match(prompt, /最新陈述优先/)
  assert.match(prompt, /客厅里刚才在讨论下周旅行/)
  assert.match(prompt, /<\/pre_wake_context>$/)
  assert.equal(buildPreWakeContextPrompt('\0  '), '')
  assert.equal(buildPreWakeContextPrompt('x'.repeat(1_300)).length < 1_700, true)
})

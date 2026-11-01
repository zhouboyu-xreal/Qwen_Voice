import assert from 'node:assert/strict'
import test from 'node:test'
import { setRuntimeLanguage } from '../src/i18n.js'
setRuntimeLanguage('zh-CN')
import { eventChanges, formatTime, recordFields, statusLabel, validityLabel } from '../src/memory/presentation.js'

test('preserves distinct intent statuses and does not infer completion from dates', () => {
  assert.equal(statusLabel({ object_type: 'goal', status: 'active' }), '推进中')
  assert.equal(statusLabel({ object_type: 'plan', status: 'occurred' }), '已发生')
  assert.equal(validityLabel({ object_type: 'work_item', status: 'open', time_end: '2020-01-01' }), '')
})

test('shows expired validity without rewriting the stored state', () => {
  const item = { state_type: 'situational_state', status: 'active', valid_to: '2020-01-01' }
  assert.equal(validityLabel(item, Date.parse('2026-11-01')), '已过有效期')
  assert.equal(item.status, 'active')
  assert.equal(validityLabel({ ...item, valid_to: '', valid_from: '2030-01-01' }), '尚未生效')
})

test('shows meaningful changes while keeping runtime IDs and metadata out of history', () => {
  const changes = eventChanges({
    previous_payload: { status: 'open', world_owner_entity_id: 1, details: { deliverable: '草稿' } },
    new_payload: { status: 'completed', world_owner_entity_id: 2, details: { deliverable: '演示材料' }, metadata: { raw: 'internal' } },
  })
  assert.deepEqual(changes.map(item => item.key), ['status', 'deliverable'])
  assert.equal(changes[0].after, '已完成')
  const fields = recordFields({ object_type: 'work_item', status: 'open', details: { action_text: '准备演示', prompt_version: 'hidden' } })
  assert.equal(fields.some(item => item.key === 'prompt_version'), false)
})


test('date-only plans do not invent a time of day', () => {
  assert.doesNotMatch(formatTime('2026-11-04'), /\d{1,2}:\d{2}/u)
  assert.doesNotMatch(formatTime('2026-11-04T00:00:00+08:00', 'day'), /\d{1,2}:\d{2}/u)
})


test('typed intent times and flattened history aliases are displayed once', () => {
  const fields = recordFields({ object_type: 'goal', time_end: '2026-11-04', details: { target_at: '2026-11-04' } })
  assert.equal(fields.filter(field => field.label === '目标时间').length, 1)
  assert.equal(eventChanges({
    previous_payload: {}, new_payload: { target_at: '2026-11-04', details: { target_at: '2026-11-04' } },
  }).filter(field => field.key === 'target_at').length, 1)
})

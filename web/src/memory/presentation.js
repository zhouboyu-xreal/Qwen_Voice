import { t } from '../i18n.js'

export const TYPES = {
  intents: { goal: '目标', plan: '计划', work_item: '待办' },
  states: { situational_state: '当前状态', recurring_state: '规律习惯', enduring_state: '长期特征' },
}
export const DOMAINS = {
  general: '通用', work: '工作', life: '生活', health: '健康', family: '家庭',
  learning: '学习', social: '社交', interest: '兴趣', travel: '出行',
  information: '信息', finance: '财务',
}
export const STATUSES = {
  active: '有效', planned: '已计划', rescheduled: '已改期', open: '待处理',
  in_progress: '进行中', blocked: '受阻', completed: '已完成', cancelled: '已取消',
  occurred: '已发生', weakened: '依据减弱', expired: '已失效',
}
const FIELDS = {
  summary: '内容', state_text: '状态描述', status: '状态', time_start: '开始时间',
  time_end: '结束或截止时间', start_at: '开始时间', end_at: '结束时间',
  due_at: '截止时间', target_at: '目标时间', valid_from: '生效时间', valid_to: '有效期至',
  desired_outcome: '期望结果', success_criteria: '达成标准', event_or_activity: '活动安排',
  time_precision: '时间精度', location_text: '地点', action_text: '具体行动',
  deliverable: '交付物', responsibility_type: '责任类型', priority: '优先级',
  completed_at: '完成时间', conditions: '适用条件', confidence: '提炼置信度',
  participants: '参与者', details: '补充信息', condition_text: '适用条件',
}
const VALUES = {
  personal_action: '个人行动', commitment: '承诺', assigned: '受指派',
  high: '高', medium: '中', low: '低', unknown: '未明确',
  exact: '精确', day: '日期', date: '日期', week: '周', month: '月',
}
const EVIDENCE = {
  fact: '提炼事实', user_claim: '用户陈述', user_activity: '用户活动',
  user_thread: '工作线索', user_thread_event: '线索变化',
  user_observed_pattern: '行为规律', user_action_signal: '行为观察',
  goal: '目标', plan: '计划', work_item: '待办',
}
const OPERATIONS = {
  created: '新增', updated: '更新', create: '新增', revise: '修订',
  reinforce: '强化', weaken: '减弱', expire: '失效',
}

export function typeLabel(item) {
  return t(TYPES.intents[item.object_type] || TYPES.states[item.state_type] || '未分类')
}
export function domainLabel(value) { return t(DOMAINS[value] || value || '未分类') }
export function statusLabel(item) {
  if (item.object_type === 'goal' && item.status === 'active') return t('推进中')
  return t(STATUSES[item.status] || item.status || '未明确')
}
export function titleFor(item) { return item.summary || item.state_text || item.state_topic || t('无标题') }
export function fieldLabel(key) { return t(FIELDS[key] || key) }
export function evidenceLabel(kind) { return t(EVIDENCE[kind] || kind) }
export function operationLabel(event) { return t(OPERATIONS[event.operation || event.event_type] || '更新') }

export function formatTime(value, precision = '') {
  if (!value) return t('未明确')
  const dateOnly = /^\d{4}-\d{2}-\d{2}$/u.test(String(value))
  const date = new Date(dateOnly ? `${value}T00:00:00` : value)
  if (!Number.isNaN(date.getTime()) && (dateOnly || ['day', 'date', 'month'].includes(precision))) {
    return date.toLocaleDateString(undefined, { year: 'numeric', month: '2-digit',
      ...(precision === 'month' ? {} : { day: '2-digit' }),
    })
  }
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString(undefined, {
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  })
}

export function validityLabel(item, now = Date.now()) {
  if (!item.state_type) return ''
  if (item.status === 'expired') return t('已失效')
  const end = Date.parse(item.valid_to)
  if (Number.isFinite(end) && end <= now) return t('已过有效期')
  const start = Date.parse(item.valid_from)
  return Number.isFinite(start) && start > now ? t('尚未生效') : ''
}

export function valueText(value) {
  if (value === null || value === undefined || value === '') return t('未明确')
  if (Array.isArray(value)) return value.map(valueText).join(' · ') || t('未明确')
  if (typeof value === 'object') {
    return Object.entries(value).map(([key, entry]) => `${fieldLabel(key)}：${valueText(entry)}`).join('；') || t('未明确')
  }
  return t(STATUSES[value] || VALUES[value] || String(value))
}

export function recordFields(item) {
  const fields = item.object_type
    ? [['status', statusLabel(item)], ['time_start', item.time_start], ['time_end', item.time_end],
        ...Object.entries(item.details || {}).filter(([key]) => Object.hasOwn(FIELDS, key)
          && !(['target_at', 'due_at', 'end_at'].includes(key) && item.time_end)
          && !(key === 'start_at' && item.time_start))]
    : [['status', statusLabel(item)], ['conditions', item.conditions], ['valid_from', item.valid_from], ['valid_to', item.valid_to]]
  return fields.filter(([, value]) => value !== null && value !== undefined && value !== ''
    && (typeof value !== 'object' || Object.keys(value).length)).map(([key, value]) => ({
    key, label: key === 'time_end'
      ? t({ goal: '目标时间', plan: '结束时间', work_item: '截止时间' }[item.object_type])
      : fieldLabel(key), value: /(?:_at|_from|_to|time_start|time_end)$/u.test(key)
      ? formatTime(value, item.details?.time_precision) : valueText(value),
  }))
}

export function eventChanges(event) {
  const before = event.previous_payload || {}
  const after = event.new_payload || {}
  const changes = Object.keys(FIELDS).filter(key => key !== 'details' && key !== 'confidence'
    && Object.hasOwn(after, key) && JSON.stringify(before[key]) !== JSON.stringify(after[key]))
    .map(key => ({ key, label: fieldLabel(key), before: valueText(before[key]), after: valueText(after[key]) }))
    .concat(Object.entries(after.details || {}).filter(([key, value]) => Object.hasOwn(FIELDS, key)
      && JSON.stringify(before.details?.[key]) !== JSON.stringify(value))
      .map(([key, value]) => ({ key, label: fieldLabel(key), before: valueText(before.details?.[key]), after: valueText(value) })))
  return [...new Map(changes.map(change => [change.key, change])).values()]
}

export function inspectionError(code) {
  return t({
    memory_inspection_unsupported: '当前记忆提供方不支持结构化查看',
    memory_record_not_found: '这条记录已不存在，请刷新列表',
    invalid_inspection_query: '筛选条件无效，请重新选择',
    access_required: '连接身份已失效，请重新连接 Gateway',
  }[code] || '读取记忆失败，请刷新重试')
}

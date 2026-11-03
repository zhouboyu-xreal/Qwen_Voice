// Structured inspection is separate from the synchronous Realtime documents.
const TYPES = {
  intents: ['goal', 'plan', 'work_item'],
  states: ['situational_state', 'recurring_state', 'enduring_state'],
  activities: [],
}
const STATUSES = new Set([
  'active', 'planned', 'rescheduled', 'open', 'in_progress', 'blocked',
  'completed', 'cancelled', 'occurred', 'weakened', 'expired',
])

function positive(value, fallback, maximum) {
  if (value === undefined) return fallback
  if (typeof value !== 'string' || !/^[1-9]\d*$/u.test(value)) return null
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed <= maximum ? parsed : null
}

function instant(value) {
  if (value === undefined || value === '') return ''
  if (typeof value !== 'string' || value.length > 40
    || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$/u.test(value)) return null
  const timestamp = Date.parse(value)
  const day = new Date(`${value.slice(0, 10)}T00:00:00Z`)
  if (!Number.isFinite(timestamp) || !Number.isFinite(day.getTime())
    || day.toISOString().slice(0, 10) !== value.slice(0, 10)) return null
  return new Date(timestamp).toISOString()
}

export function memoryInspectionOptions(section, id, query = {}) {
  if (!Object.hasOwn(TYPES, section)) return null
  const options = { section }
  for (const [key, fallback, maximum] of [
    ['page', 1, 1_000_000], ['pageSize', 20, 100],
    ['evidencePage', 1, 1_000_000], ['historyPage', 1, 1_000_000],
  ]) {
    options[key] = positive(query[key], fallback, maximum)
    if (options[key] === null) return null
  }
  if (id !== undefined) {
    options.id = positive(id, null, Number.MAX_SAFE_INTEGER)
    if (options.id === null) return null
  }
  if (section === 'activities') {
    if (query.role !== undefined && typeof query.role !== 'string') return null
    options.role = query.role || ''
    if (options.role && !['actor', 'participant', 'contextual'].includes(options.role)) return null
    options.from = instant(query.from)
    options.to = instant(query.to)
    options.sourcePage = positive(query.sourcePage, 1, 1_000_000)
    if (options.from === null || options.to === null || options.sourcePage === null) return null
    if (options.from && options.to && Date.parse(options.from) >= Date.parse(options.to)) return null
    return options
  }
  for (const key of ['type', 'scope', 'status', 'domain']) {
    if (query[key] !== undefined && typeof query[key] !== 'string') return null
    options[key] = query[key] || ''
  }
  options.scope ||= 'current'
  if (options.type && !TYPES[section].includes(options.type)) return null
  if (!['current', 'history', 'all'].includes(options.scope)) return null
  if (options.status && !STATUSES.has(options.status)) return null
  if (options.domain.length > 80) return null
  // ownerId and database paths are deliberately never accepted from a client.
  return options
}

export function registerMemoryInspectionRoutes(app, { memory }) {
  const inspect = async (req, res) => {
    res.setHeader('cache-control', 'no-store')
    const options = memoryInspectionOptions(req.params.section, req.params.id, req.query)
    if (!options) return res.status(400).json({ code: 'invalid_inspection_query' })
    if (typeof memory?.inspect !== 'function') {
      return res.status(501).json({ code: 'memory_inspection_unsupported', supported: false })
    }
    try {
      const result = await memory.inspect(req.identity.ownerId, options)
      if (result?.supported === false) {
        return res.status(501).json({ ...result, code: 'memory_inspection_unsupported' })
      }
      if (options.id && result?.available && !result.item) {
        return res.status(404).json({ code: 'memory_record_not_found' })
      }
      return res.json(result)
    } catch {
      // Paths, credentials and raw sidecar stderr must not become UI messages.
      return res.status(503).json({ code: 'memory_inspection_unavailable' })
    }
  }
  app.get('/api/pwm/:section', inspect)
  app.get('/api/pwm/:section/:id', inspect)
}

const MAX_BYTES = 512 * 1024
const ID = /^[a-f0-9]{64}$/u

export function feishuDocument(body, ownerRequired = false) {
  const { filename, content, ownerName } = body || {}
  if (typeof filename !== 'string' || filename.length > 240 || !/\.md$/iu.test(filename)
    || /[/\\\0]/u.test(filename) || typeof content !== 'string' || !content.trim()
    || content.includes('\0') || Buffer.byteLength(content) > MAX_BYTES) return null
  if (ownerRequired && (typeof ownerName !== 'string' || !ownerName.trim() || ownerName.length > 120)) return null
  return { filename, content, ...(ownerRequired ? { ownerName } : {}) }
}

export function registerFeishuImportRoutes(app, { memory }) {
  const handle = operation => async (req, res) => {
    res.setHeader('cache-control', 'no-store')
    const input = operation === 'status'
      ? ID.test(req.params.id) ? { id: req.params.id } : null
      : feishuDocument(req.body, operation === 'import')
    if (!input) return res.status(400).json({ code: 'invalid_feishu_document' })
    if (typeof memory?.feishuMeeting !== 'function') return res.status(501).json({ code: 'feishu_import_unsupported' })
    try {
      const result = await memory.feishuMeeting(req.identity.ownerId, operation, input)
      if (result?.supported === false) return res.status(501).json({ code: 'feishu_import_unsupported' })
      if (result?.ok === false) {
        const statuses = { invalid_feishu_document: 400, feishu_import_busy: 409, feishu_import_not_found: 404, memory_resetting: 409, stale_memory_generation: 409 }
        const code = Object.hasOwn(statuses, result.code) ? result.code : 'feishu_import_unavailable'
        return res.status(statuses[code] || 503).json({ code })
      }
      if (!result || result.ok !== true) throw new Error('Invalid import response')
      return res.status(operation === 'import' && !result.duplicate ? 202 : 200).json(result)
    } catch {
      return res.status(503).json({ code: 'feishu_import_unavailable' })
    }
  }
  app.post('/api/pwm/imports/feishu/preview', handle('preview'))
  app.post('/api/pwm/imports/feishu', handle('import'))
  app.get('/api/pwm/imports/feishu/:id', handle('status'))
}

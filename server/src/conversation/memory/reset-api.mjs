export function registerMemoryResetRoutes(app, { memory }) {
  const handle = operation => async (req, res) => {
    res.setHeader('cache-control', 'no-store')
    if (operation === 'start' && req.body?.confirmation !== 'clear-memory') {
      return res.status(400).json({ code: 'memory_reset_confirmation_required' })
    }
    if (typeof memory?.resetMemory !== 'function') return res.status(501).json({ code: 'memory_reset_unsupported' })
    try {
      const result = await memory.resetMemory(req.identity.ownerId, operation)
      if (result?.supported === false) return res.status(501).json({ code: 'memory_reset_unsupported' })
      if (result?.ok === false && result.code === 'memory_reset_busy') return res.status(409).json({ code: result.code })
      if (result?.ok !== true) throw new Error('Invalid reset response')
      return res.status(operation === 'start' ? 202 : 200).json(result)
    } catch {
      return res.status(503).json({ code: 'memory_reset_unavailable' })
    }
  }
  app.post('/api/pwm/reset', handle('start'))
  app.get('/api/pwm/reset', handle('status'))
}

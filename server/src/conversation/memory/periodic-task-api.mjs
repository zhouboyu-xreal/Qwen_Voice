import { isPwmPeriodicTask } from '../../../../shared/pwm-periodic-tasks.mjs'

export function registerMemoryPeriodicTaskRoutes(app, { memory }) {
  app.post('/api/memory/periodic-task', async (req, res) => {
    res.setHeader('cache-control', 'no-store')
    const taskId = req.body?.taskId
    if (!isPwmPeriodicTask(taskId)) {
      return res.status(400).json({ code: 'invalid_periodic_task' })
    }
    if (typeof memory?.triggerPeriodicTask !== 'function') {
      return res.status(501).json({ supported: false, code: 'periodic_tasks_unsupported' })
    }
    try {
      // Identity comes from Gateway authentication, never from the request body.
      const result = await memory.triggerPeriodicTask(req.identity.ownerId, taskId)
      if (result.supported === false) return res.status(501).json(result)
      return res.status(result.queued ? 202 : 200).json(result)
    } catch {
      return res.status(503).json({ code: 'periodic_task_unavailable' })
    }
  })
}

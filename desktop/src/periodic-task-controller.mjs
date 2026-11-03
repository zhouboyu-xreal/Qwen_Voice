// Host clock only: task data, eligibility, and execution belong to the receiver.
export function createPeriodicTaskController({
  tasks,
  onTrigger,
  onError = () => {},
  now = () => performance.now(),
  setTimer = setTimeout,
  clearTimer = clearTimeout,
}) {
  const ids = new Set()
  const schedules = tasks.map(({ taskId, intervalMs }) => {
    if (!taskId || ids.has(taskId) || !Number.isFinite(intervalMs) || intervalMs <= 0) {
      throw new TypeError('Periodic tasks require unique ids and positive finite intervals')
    }
    ids.add(taskId)
    return { taskId, intervalMs, nextAt: 0 }
  })
  let running = false
  let timer = null
  let abort = null
  const clear = () => {
    if (timer !== null) clearTimer(timer)
    timer = null
  }
  const arm = () => {
    clear()
    if (!running || !schedules.length) return
    const delay = Math.max(0, Math.min(...schedules.map(task => task.nextAt)) - now())
    timer = setTimer(check, Math.min(delay, 2_147_483_647))
    timer?.unref?.()
  }
  const check = () => {
    if (!running) return
    const current = now()
    for (const task of schedules) {
      if (!running) break
      if (current < task.nextAt) continue
      // Skip missed cycles after a delayed timer/resume; emit at most once.
      task.nextAt += (Math.floor((current - task.nextAt) / task.intervalMs) + 1) * task.intervalMs
      const signal = abort.signal
      try {
        Promise.resolve(onTrigger({ taskId: task.taskId, signal }))
          .catch(error => { if (!signal.aborted) onError(error, task.taskId) })
      } catch (error) {
        onError(error, task.taskId)
      }
    }
    arm()
  }
  return {
    start() {
      if (running) return
      running = true
      abort = new AbortController()
      const startedAt = now()
      for (const task of schedules) task.nextAt = startedAt + task.intervalMs
      arm()
    },
    stop() {
      running = false
      clear()
      abort?.abort()
    },
    check,
  }
}

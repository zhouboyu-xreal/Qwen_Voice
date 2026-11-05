import { PWM_USER_STATE_TASK } from '../../shared/pwm-periodic-tasks.mjs'

export function desktopPwmPeriodicTasks(env = process.env) {
  const configured = env.QWEN_AUDIO_PWM_USER_STATE_INTERVAL_SECONDS
  const seconds = configured == null || String(configured).trim() === '' ? 1200 : Number(configured)
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new TypeError('QWEN_AUDIO_PWM_USER_STATE_INTERVAL_SECONDS must be nonnegative')
  }
  return seconds === 0 ? [] : [{ taskId: PWM_USER_STATE_TASK, intervalMs: seconds * 1000 }]
}

export async function sendPwmPeriodicSignal({
  origin, accessToken = '', taskId, signal, fetchImpl = fetch,
}) {
  if (!origin) throw new Error('Gateway is not ready for periodic task signals')
  const response = await fetchImpl(`${origin.replace(/\/$/u, '')}/api/memory/periodic-task`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
    body: JSON.stringify({ taskId }),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000),
  })
  if (!response.ok) throw new Error(`PWM periodic signal returned HTTP ${response.status}`)
  return response.json()
}

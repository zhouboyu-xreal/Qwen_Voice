export const PWM_USER_STATE_TASK = 'user_state_from_claims'

export function isPwmPeriodicTask(taskId) {
  return taskId === PWM_USER_STATE_TASK
}

import { t } from '../i18n.js'

export default function AmbientRecordingButton({ state, onToggle, children }) {
  const busy = state === 'starting' || state === 'finalizing'
  const label = state === 'recording'
    ? t('停止环境录音')
    : state === 'starting'
      ? t('正在启动环境录音')
      : state === 'finalizing'
        ? t('正在处理环境录音') : t('开始环境录音')

  return (
    <button
      className={['starting', 'recording'].includes(state) ? 'active recording' : ''}
      // Keep the pointer gesture out of the orb's drag handler. Activation
      // belongs only to click: IPC can finish between pointerdown and click,
      // so toggling in both handlers can immediately stop a new recording.
      onPointerDown={event => event.stopPropagation()}
      onClick={event => {
        event.stopPropagation()
        if (!busy) onToggle()
      }}
      aria-label={label}
      aria-pressed={state === 'recording'}
      aria-busy={busy}
      title={label}
      disabled={busy}
    >
      {children}
    </button>
  )
}

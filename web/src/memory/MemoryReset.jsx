import { useEffect, useRef, useState } from 'react'
import { gatewayFetch } from '../gateway-transport.js'
import { t } from '../i18n.js'

const MESSAGES = {
  memory_reset_busy: '请先停止环境录音并等待转写结束，或等待飞书会议导入完成，再清空记忆。',
  memory_reset_unsupported: '当前记忆提供方不支持清空记忆。',
  memory_reset_unavailable: '暂时无法连接记忆服务，正在确认后台状态，请稍后查看。',
}
const PHASES = {
  waiting: '正在等待后台任务结束…', clearing: '正在清理记忆数据…', initializing: '正在初始化新的记忆库…',
  completed: '记忆已清空', failed: '清空未完成，记忆写入已暂停。请检查后台日志后重新清空。',
  interrupted: '上次清空被中断，记忆写入已暂停。请重新清空以恢复使用。',
}
async function request(start = false) {
  const response = await gatewayFetch('api/pwm/reset', {
    ...(start ? { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ confirmation: 'clear-memory' }) } : {}), cache: 'no-store',
  })
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw Object.assign(new Error(t(MESSAGES[result.code] || MESSAGES.memory_reset_unavailable)), { code: result.code })
  return result
}

export default function MemoryReset({ onCompleted, onBlocked }) {
  const dialog = useRef(null)
  const completed = useRef('')
  const sequence = useRef(0)
  const [open, setOpen] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [job, setJob] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [uncertain, setUncertain] = useState(false)
  const [error, setError] = useState('')
  const [statusError, setStatusError] = useState('')
  const [revision, setRevision] = useState(0)
  const blocked = submitting || uncertain || Boolean(job && job.status !== 'completed')
  const working = submitting || uncertain || ['queued', 'running'].includes(job?.status)

  useEffect(() => { onBlocked(blocked) }, [blocked, onBlocked])
  useEffect(() => {
    if (open) dialog.current?.showModal()
    else dialog.current?.close()
  }, [open])
  useEffect(() => {
    let stopped = false, timer
    const current = sequence.current
    async function poll() {
      try {
        const result = await request()
        if (stopped || current !== sequence.current) return
        setJob(result.job); setUncertain(false); setStatusError('')
        if (result.job?.status === 'completed' && completed.current !== result.job.id) {
          completed.current = result.job.id
          onCompleted()
        }
      } catch (failure) {
        if (stopped || current !== sequence.current) return
        setStatusError(failure.message)
      }
      if (!stopped) timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [revision, onCompleted])

  async function submit() {
    if (!confirmed || working) return
    sequence.current += 1
    setSubmitting(true); setError('')
    try {
      const result = await request(true)
      setJob(result.job); setConfirmed(false)
    } catch (failure) {
      setError(failure.message)
      if (!['memory_reset_busy', 'memory_reset_unsupported'].includes(failure.code)) setUncertain(true)
    } finally {
      setSubmitting(false); setRevision(value => value + 1)
    }
  }

  return <>
    <button type="button" className="memory-danger" onClick={() => setOpen(true)}>{t(blocked ? '查看清空状态' : '清空记忆')}</button>
    <dialog className="memory-import-dialog memory-reset-dialog" ref={dialog} aria-labelledby="memory-reset-title"
      onCancel={event => { event.preventDefault(); setOpen(false) }}>
      <div className="memory-import-heading"><h2 id="memory-reset-title">{t('清空记忆')}</h2>
        <button type="button" onClick={() => setOpen(false)}>{t('关闭')}</button></div>
      <p>{t('将永久删除当前用户的活动、意图与安排、状态及其证据，同时清理待处理任务、缓存、飞书导入归档与去重记录。此操作不可恢复。')}</p>
      <p>{t('原始环境录音、您选择的源会议文件、模型配置和本人声音样本会保留。清空后可以重新导入同一份会议记录。')}</p>
      <p>{t('请先停止环境录音并等待转写结束，结束语音对话，并等待飞书导入完成。清空期间暂停记忆写入，完成后请重新开启语音或录音。')}</p>
      {!working && <>
        <label className="memory-reset-confirm"><input type="checkbox" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} />
          {t('我了解此操作不可恢复，确认清空当前用户的记忆。')}</label>
        <button type="button" className="memory-danger" disabled={!confirmed} onClick={submit}>{t('确认清空记忆')}</button>
      </>}
      {(job || uncertain || submitting) && <section className="memory-import-result" role="status">
        <p>{t(uncertain ? '正在确认后台状态…' : PHASES[job?.phase] || '正在等待后台任务结束…')}</p>
        {working && <p>{t('可以关闭此窗口，后台会继续处理。')}</p>}
      </section>}
      {(error || statusError) && <p className="memory-error" role="alert">{error || statusError}</p>}
    </dialog>
  </>
}

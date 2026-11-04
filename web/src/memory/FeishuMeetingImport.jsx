import { useEffect, useRef, useState } from 'react'
import { gatewayFetch } from '../gateway-transport.js'
import { t } from '../i18n.js'
import { formatTime } from './presentation.js'

const STORAGE = 'qwen.feishuImportJob'
const TERMINAL = new Set(['completed', 'partial', 'failed', 'interrupted'])
const ERRORS = {
  invalid_feishu_document: '无法解析此文件，请选择包含会议日期、时间及说话人时间戳的飞书 Markdown 文字记录，并选择正确的本人姓名。',
  memory_resetting: '记忆正在清空或等待恢复，请先完成清空。',
  stale_memory_generation: '记忆版本已更新，请重新尝试导入。',
  feishu_import_busy: '请先结束环境录音，并等待已有导入完成后重试。',
  feishu_import_not_found: '导入任务不存在，请重新选择文件。',
  feishu_import_unsupported: '当前记忆提供方不支持飞书会议导入。',
  feishu_import_unavailable: '暂时无法导入，请检查后台服务后重试。',
}
const PHASES = { queued: '等待导入', preparing: '正在准备导入…', extracting: '正在提炼会议记忆…',
  updating_states: '正在更新我的状态…', completed: '导入完成', failed: '导入未完成', interrupted: '导入已中断' }

async function request(path, body, signal) {
  const response = await gatewayFetch(`api/pwm/imports/feishu${path}`, {
    ...(body ? { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) } : {}),
    cache: 'no-store', signal,
  })
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw Object.assign(new Error(t(ERRORS[result.code] || ERRORS.feishu_import_unavailable)), { code: result.code })
  return result
}
function remember(id) {
  try { if (id) sessionStorage.setItem(STORAGE, id); else sessionStorage.removeItem(STORAGE) } catch { /* Optional reload recovery. */ }
}

export default function FeishuMeetingImport({ onCompleted, disabled = false }) {
  const dialog = useRef(null)
  const [open, setOpen] = useState(false)
  const [document, setDocument] = useState(null)
  const [preview, setPreview] = useState(null)
  const [ownerName, setOwnerName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [jobId, setJobId] = useState('')
  const [job, setJob] = useState(null)
  const working = Boolean(jobId && (!job || !TERMINAL.has(job.status)))

  useEffect(() => {
    try { setJobId(sessionStorage.getItem(STORAGE) || '') } catch { /* Storage can be disabled. */ }
  }, [])
  useEffect(() => {
    if (open) dialog.current?.showModal()
    else dialog.current?.close()
  }, [open])
  useEffect(() => {
    if (!jobId) return undefined
    const controller = new AbortController()
    let timer
    async function poll() {
      let done = false
      try {
        const result = await request(`/${jobId}`, null, controller.signal)
        if (controller.signal.aborted) return
        setJob(current => ({ ...result, duplicate: current?.duplicate }))
        done = TERMINAL.has(result.status)
        if (done) { remember(''); onCompleted?.() }
      } catch (failure) {
        if (controller.signal.aborted) return
        setError(failure.message)
        if (failure.code === 'feishu_import_not_found') { done = true; setJobId(''); remember('') }
      }
      if (!done && !controller.signal.aborted) timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [jobId, onCompleted])

  async function chooseFile(event) {
    const file = event.target.files?.[0]
    if (!file || working) return
    setBusy(true); setError(''); setPreview(null); setDocument(null); setOwnerName(''); setJob(null); setJobId(''); remember('')
    try {
      if (!/\.md$/iu.test(file.name) || file.size > 512 * 1024) throw new Error(t('请选择不超过 512 KB 的单个 Markdown（.md）文件。'))
      const input = { filename: file.name, content: await file.text() }
      const result = await request('/preview', input)
      setDocument(input); setPreview(result)
    } catch (failure) { setError(failure.message) }
    finally { setBusy(false) }
  }
  async function submit() {
    if (!document || !ownerName || busy || working) return
    setBusy(true); setError('')
    try {
      const result = await request('', { ...document, ownerName })
      setJob(result); setJobId(result.id); remember(result.id)
    } catch (failure) { setError(failure.message) }
    finally { setBusy(false) }
  }
  const jobLabel = job?.status === 'partial' ? t('部分提炼失败，请查看后台日志；已保存的结果会保留。')
    : job?.status === 'interrupted' ? t('导入曾中断，可能已有部分记忆，请先检查结果，系统不会自动重复导入。')
      : job?.status === 'failed' ? t('导入未完成，请查看后台日志。') : t(PHASES[job?.phase] || '等待导入')

  return <>
    <button type="button" disabled={disabled} onClick={() => setOpen(true)}>{working ? t('查看导入进度') : t('导入飞书会议')}</button>
    <dialog className="memory-import-dialog" ref={dialog} aria-labelledby="memory-import-title"
      onCancel={event => { event.preventDefault(); setOpen(false) }}>
      <div className="memory-import-heading"><h2 id="memory-import-title">{t('导入飞书会议')}</h2>
        <button type="button" onClick={() => setOpen(false)}>{t('关闭')}</button></div>
      <p>{t('选择单个飞书 Markdown 文字记录，导入后提炼并保存到当前用户的记忆。')}</p>
      <p>{t('请先停止环境录音，导入期间暂停语音对话。')}</p>
      <label className="memory-import-file">{t('会议记录文件')}
        <input type="file" accept=".md,text/markdown" onChange={chooseFile} disabled={busy || working} /></label>
      {preview && <>
        <dl className="memory-import-summary">
          <dt>{t('会议主题')}</dt><dd>{preview.meeting_title}</dd>
          <dt>{t('会议开始时间')}</dt><dd>{formatTime(preview.session_start)}</dd>
          <dt>{t('文字片段')}</dt><dd>{preview.source_segment_count}</dd>
        </dl>
        <label className="memory-import-owner">{t('会议中的本人姓名')}
          <select value={ownerName} disabled={busy || working || Boolean(job)} onChange={event => setOwnerName(event.target.value)}>
            <option value="">{t('请选择本人姓名')}</option>
            {preview.speakers.map(name => <option key={name} value={name}>{name}</option>)}
          </select></label>
        <p>{t('其他参会人的发言也会保留，用于理解会议上下文。')}</p>
        <button type="button" disabled={!ownerName || busy || working || Boolean(job)} onClick={submit}>{t('导入并提炼')}</button>
      </>}
      {busy && <p role="status">{t('正在处理…')}</p>}
      {job && <section className="memory-import-result" role="status">
        <b>{job.filename}</b><p>{jobLabel}</p>
        {job.status === 'completed' && <p>{t('较早的会议可切换至“历史记录”或“全部记录”查看提炼结果。')}</p>}
        {job.ownerName && <p>{t('本人姓名')}：{job.ownerName}</p>}
        {job.duplicate && <p>{t('该文件已导入或正在处理，本次不会重复写入。')}</p>}
        {working && <p>{t('可以关闭此窗口，后台会继续处理。')}</p>}
      </section>}
      {error && <p className="memory-error" role="alert">{error}</p>}
    </dialog>
  </>
}

import { useCallback, useEffect, useRef, useState } from 'react'
import { gatewayFetch } from '../gateway-transport.js'
import { t } from '../i18n.js'
import { memoryListQuery } from './inspection-query.js'
import {
  ACTIVITY_ROLES, SECTIONS, DOMAINS, STATUSES, TYPES, activityRoleLabel, domainLabel, eventChanges, evidenceLabel,
  formatTime, inspectionError, operationLabel, recordFields, statusLabel,
  titleFor, typeLabel, validityLabel, isActivity, observationWindow,
} from './presentation.js'
import FeishuMeetingImport from './FeishuMeetingImport.jsx'
import './memory-center.css'

const PAGE_SIZE = 20
const POLL_MS = 8000

async function read(path, signal) {
  const response = await gatewayFetch(path, { cache: 'no-store', signal })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw Object.assign(new Error(inspectionError(payload.code)), { code: payload.code })
  return payload
}

function Pager({ page, total, pageSize = PAGE_SIZE, onChange, disabled, label }) {
  const pages = Math.max(1, Math.ceil(total / pageSize))
  if (pages === 1) return null
  return <nav className="memory-pager" aria-label={label || t('记录分页')}>
    <button type="button" disabled={disabled || page <= 1} onClick={() => onChange(page - 1)}>{t('上一页')}</button>
    <span>{t('{page} / {pages} 页', { page, pages })}</span>
    <button type="button" disabled={disabled || page >= pages} onClick={() => onChange(page + 1)}>{t('下一页')}</button>
  </nav>
}

export function MemoryRecordDetail({ data, evidencePage = 1, historyPage = 1, sourcePage = 1,
  setEvidencePage, setHistoryPage, setSourcePage, sourceExpanded, setSourceExpanded, busy }) {
  const item = data.item
  if (!item) return <p className="memory-empty">{t('这条记录已不存在，请刷新列表')}</p>
  const validity = validityLabel(item)
  const activity = isActivity(item)
  return <>
    <div className="memory-record-meta">{typeLabel(item)} · {activity
      ? activityRoleLabel(item.user_role) : domainLabel(item.primary_domain || item.state_domain)}</div>
    <h2>{titleFor(item)}</h2>
    {activity && <p className="memory-muted">{t('活动描述基于采集窗口观察；观察时间窗口不代表精确活动时长。')}</p>}
    {validity && <p className="memory-validity">{validity} · {t('存储状态')}：{statusLabel(item)}</p>}
    <dl className="memory-fields">
      {item.entity_name && <><dt>{t('涉及人物')}</dt><dd>{item.entity_name}</dd></>}
      {recordFields(item).map(field => <div key={field.key}><dt>{field.label}</dt><dd>{field.value}</dd></div>)}
      <div><dt>{t(activity ? '提炼保存时间' : '最近更新')}</dt><dd>{formatTime(activity ? item.created_at : item.updated_at)}</dd></div>
      {item.last_observed_at && <div><dt>{t('最近观察')}</dt><dd>{formatTime(item.last_observed_at)}</dd></div>}
      <div><dt>{t('提炼置信度')}</dt><dd>{Number.isFinite(item.confidence) ? item.confidence.toFixed(2) : t('未明确')}</dd></div>
    </dl>
    <section className="memory-detail-section">
      <h3>{t(activity ? '关联事实' : '依据')} <span>({data.evidence?.total || 0})</span></h3>
      {!data.evidence?.items?.length && <p className="memory-muted">{t('未保存关联依据')}</p>}
      {data.evidence?.items?.map(evidence => <div className="memory-evidence" key={`${evidence.evidence_type}:${evidence.evidence_id}:${evidence.role}`}>
        <div className="memory-record-meta">{evidenceLabel(evidence.evidence_type)} · #{evidence.evidence_id}</div>
        <p>{evidence.resolved ? evidence.text || t('来源未保存正文') : t('来源记录已不可用')}</p>
        {evidence.occurred_at && <small>{formatTime(evidence.occurred_at)}</small>}
        {!activity && (evidence.source_segments?.length ? <details>
          <summary>{t('原始片段摘录（最多 6 段）')}</summary>
          {evidence.source_segments.map(segment => <blockquote key={segment.id}>
            <p>{segment.text}</p><small>{formatTime(segment.started_at)}</small>
          </blockquote>)}
        </details> : evidence.resolved && <small>{t('未保存可追溯的原始片段')}</small>)}
      </div>)}
      <Pager page={evidencePage} total={data.evidence?.total || 0} pageSize={data.evidence?.pageSize}
        onChange={setEvidencePage} disabled={busy} label={t('依据分页')} />
    </section>
    {activity && <section className="memory-detail-section">
      <h3>{t('来源窗口')} <span>({data.sources?.total || 0})</span></h3>
      <p className="memory-muted">{t('来源窗口可能包含其他话题，摘录用于查看上下文。')}</p>
      {!data.sources?.items?.length && <p className="memory-muted">{t('未保存可追溯的原始片段')}</p>}
      {!!data.sources?.items?.length && <details className="memory-source-window" open={sourceExpanded}
        onToggle={event => setSourceExpanded?.(event.currentTarget.open)}>
        <summary>{t('查看来源窗口摘录（每页最多 6 段）')}</summary>
        {data.sources.items.map(segment => <blockquote key={segment.id}>
          <p>{segment.resolved ? segment.text || t('来源未保存正文') : t('来源记录已不可用')}</p>
          {segment.resolved && <small>{formatTime(segment.started_at)}</small>}
        </blockquote>)}
        <Pager page={sourcePage} total={data.sources.total} pageSize={data.sources.pageSize}
          onChange={setSourcePage} disabled={busy} label={t('来源窗口分页')} />
      </details>}
    </section>}
    {!activity && <section className="memory-detail-section">
      <h3>{t('变更记录')} <span>({data.history?.total || 0})</span></h3>
      {!data.history?.items?.length && <p className="memory-muted">{t('未保存变更记录')}</p>}
      {data.history?.items?.map(event => <div className="memory-event" key={event.id}>
        <header><b>{operationLabel(event)}</b><time>{formatTime(event.effective_at || event.created_at)}</time></header>
        {event.reason && <p>{event.reason}</p>}
        {eventChanges(event).map(change => <div className="memory-change" key={change.key}>
          <span>{change.label}</span><p>{change.before} → {change.after}</p>
        </div>)}
      </div>)}
      <Pager page={historyPage} total={data.history?.total || 0} pageSize={data.history?.pageSize}
        onChange={setHistoryPage} disabled={busy} label={t('变更记录分页')} />
    </section>}
  </>
}

export default function MemoryCenter() {
  const [section, setSection] = useState('intents')
  const [type, setType] = useState('')
  const [scope, setScope] = useState('current')
  const [status, setStatus] = useState('')
  const [domain, setDomain] = useState('')
  const [role, setRole] = useState('')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState(null)
  const [evidencePage, setEvidencePage] = useState(1)
  const [historyPage, setHistoryPage] = useState(1)
  const [sourcePage, setSourcePage] = useState(1)
  const [sourceExpanded, setSourceExpanded] = useState(false)
  const [revision, setRevision] = useState(0)
  const [list, setList] = useState(null)
  const [listLoading, setListLoading] = useState(true)
  const [listError, setListError] = useState('')
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')
  const pending = useRef({ list: false, detail: false })
  const activities = section === 'activities'
  const listKey = `${section}:${type}:${scope}:${status}:${domain}:${role}:${fromDate}:${toDate}:${page}`
  const detailKey = `${section}:${selected}:${evidencePage}:${historyPage}:${sourcePage}`
  const refresh = useCallback(() => setRevision(value => value + 1), [])

  useEffect(() => {
    const refreshVisible = () => {
      if (document.visibilityState !== 'hidden' && !pending.current.list && !pending.current.detail) refresh()
    }
    const interval = setInterval(refreshVisible, POLL_MS)
    document.addEventListener('visibilitychange', refreshVisible)
    return () => {
      clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshVisible)
    }
  }, [refresh])

  useEffect(() => {
    if (document.visibilityState === 'hidden') return undefined
    let query
    try {
      query = memoryListQuery({ section, type, scope, status, domain, role, fromDate, toDate, page, pageSize: PAGE_SIZE })
    } catch {
      setListLoading(false)
      setListError(t('请选择有效的日期范围，开始日期不能晚于结束日期'))
      setSelected(null)
      return undefined
    }
    const controller = new AbortController()
    const requests = pending.current
    requests.list = true
    setListLoading(true)
    setListError('')
    void read(`api/pwm/${section}?${query}`, controller.signal).then(data => {
      if (controller.signal.aborted) return
      const lastPage = Math.max(1, Math.ceil(data.total / PAGE_SIZE))
      if (page > lastPage) { setPage(lastPage); return }
      setList({ key: listKey, data })
      setSelected(current => data.items?.some(item => item.id === current) ? current : data.items?.[0]?.id || null)
    }).catch(error => {
      if (!controller.signal.aborted) setListError(error.message)
    }).finally(() => {
      if (!controller.signal.aborted) { requests.list = false; setListLoading(false) }
    })
    return () => { controller.abort(); requests.list = false }
  }, [section, type, scope, status, domain, role, fromDate, toDate, page, listKey, revision])

  useEffect(() => {
    setEvidencePage(1)
    setHistoryPage(1)
    setSourcePage(1)
    setSourceExpanded(false)
  }, [section, selected])

  useEffect(() => {
    if (!selected) {
      setDetailLoading(false)
      setDetailError('')
      return undefined
    }
    if (document.visibilityState === 'hidden') return undefined
    const controller = new AbortController()
    const requests = pending.current
    requests.detail = true
    setDetailLoading(true)
    setDetailError('')
    const query = new URLSearchParams({ evidencePage: String(evidencePage), historyPage: String(historyPage) })
    if (section === 'activities') query.set('sourcePage', String(sourcePage))
    void read(`api/pwm/${section}/${selected}?${query}`, controller.signal).then(data => {
      if (!controller.signal.aborted) setDetail({ key: detailKey, data })
    }).catch(error => {
      if (!controller.signal.aborted) setDetailError(error.message)
    }).finally(() => {
      if (!controller.signal.aborted) { requests.detail = false; setDetailLoading(false) }
    })
    return () => { controller.abort(); requests.detail = false }
  }, [section, selected, evidencePage, historyPage, sourcePage, detailKey, revision])

  const data = list?.key === listKey ? list.data : null
  const detailData = detail?.key === detailKey ? detail.data : null
  const items = data?.items || []
  const resetFilter = (setter, value) => { setter(value); setPage(1); setSelected(null); refresh() }
  const changeSection = next => {
    if (next === section) return
    setSection(next); setType(''); setStatus(''); setDomain(''); setScope('current')
    setRole(''); setFromDate(''); setToDate(''); setPage(1); setSelected(null)
    setList(null); setDetail(null)
  }

  return <main className="memory-center">
    <header className="memory-titlebar">
      <div><h1>{t('我的记忆')}</h1><span>Qwen Audio · PersonalWorldModel</span></div>
      <div className="memory-refresh"><FeishuMeetingImport onCompleted={refresh} /><small>{data?.fetchedAt ? t('最近读取：{time}', { time: formatTime(data.fetchedAt) }) : t('等待读取')}</small>
        <button type="button" onClick={refresh} disabled={listLoading || detailLoading}>{t('刷新')}</button></div>
    </header>
    <div className="memory-shell">
      <nav className="memory-navigation" aria-label={t('记忆类别')}>
        {Object.entries(SECTIONS).map(([key, label]) => <button type="button" key={key}
          aria-pressed={section === key} onClick={() => changeSection(key)}>{t(label)}</button>)}
        <p>{t('查看记忆不会激活语音')}</p>
      </nav>
      <section className="memory-content" aria-label={t(SECTIONS[section])}>
        <div className="memory-toolbar">
          {!activities && <div className="memory-type-tabs" role="group" aria-label={t('记录类型')}>
            <button type="button" aria-pressed={!type} onClick={() => resetFilter(setType, '')}>{t('全部类型')}</button>
            {Object.entries(TYPES[section]).map(([key, label]) => <button type="button" key={key}
              aria-pressed={type === key} onClick={() => resetFilter(setType, key)}>{t(label)}</button>)}
          </div>}
          <div className="memory-filters">
            {activities ? <>
              <label>{t('参与角色')}<select aria-label={t('参与角色')} value={role} onChange={event => resetFilter(setRole, event.target.value)}>
                <option value="">{t('全部角色')}</option>
                {Object.entries(ACTIVITY_ROLES).map(([key, label]) => <option value={key} key={key}>{t(label)}</option>)}
              </select></label>
              <label>{t('开始日期')}<input aria-label={t('开始日期')} type="date" value={fromDate}
                onInput={event => resetFilter(setFromDate, event.currentTarget.value)} /></label>
              <label>{t('结束日期')}<input aria-label={t('结束日期')} type="date" value={toDate}
                onInput={event => resetFilter(setToDate, event.currentTarget.value)} /></label>
            </> : <>
            <label>{t('显示范围')}<select aria-label={t('显示范围')} value={scope} onChange={event => resetFilter(setScope, event.target.value)}>
              <option value="current">{t('当前记录')}</option><option value="history">{t('历史记录')}</option><option value="all">{t('全部记录')}</option>
            </select></label>
            <label>{t('状态')}<select aria-label={t('状态')} value={status} onChange={event => resetFilter(setStatus, event.target.value)}>
              <option value="">{t('全部状态')}</option>
              {Object.entries(STATUSES).filter(([key]) => section === 'states'
                ? ['active', 'weakened', 'expired'].includes(key) : !['weakened', 'expired'].includes(key))
                .map(([key, label]) => <option value={key} key={key}>{t(label)}</option>)}
            </select></label>
            <label>{t('领域')}<select aria-label={t('领域')} value={domain} onChange={event => resetFilter(setDomain, event.target.value)}>
              <option value="">{t('全部领域')}</option>
              {Object.entries(DOMAINS).map(([key, label]) => <option value={key} key={key}>{t(label)}</option>)}
            </select></label>
            </>}
          </div>
        </div>
        {listError && <div className="memory-error" role="alert">{listError}</div>}
        {data?.available === false && <p className="memory-empty">{t(data.reason === 'not_created'
          ? '当前用户尚未生成这类记忆，提炼并保存后会显示在这里'
          : '当前数据库结构不支持查看，请更新 Agent Memory')}</p>}
        <div className="memory-columns">
          <div className="memory-list-pane" aria-busy={listLoading}>
            <div className="memory-list-count" aria-live="polite">{listLoading && !data ? t('正在读取…') : t('{count} 条记录', { count: data?.total || 0 })}</div>
            <div className="memory-list">
              {!listLoading && !listError && data?.available && !items.length && <p className="memory-empty">{t('没有符合筛选条件的记录')}</p>}
              {items.map(item => <button type="button" className="memory-list-item" aria-pressed={selected === item.id}
                key={item.id} onClick={() => setSelected(item.id)}>
                <span className="memory-record-meta">{typeLabel(item)} · {isActivity(item) ? activityRoleLabel(item.user_role) : statusLabel(item)}</span>
                <strong>{titleFor(item)}</strong>
                <span>{isActivity(item) ? observationWindow(item)
                  : <>{domainLabel(item.primary_domain || item.state_domain)} · {formatTime(item.time_end || item.last_observed_at || item.updated_at, item.time_end ? item.details?.time_precision : '')}</>}</span>
                {validityLabel(item) && <span className="memory-validity">{validityLabel(item)}</span>}
              </button>)}
            </div>
            <Pager page={page} total={data?.total || 0} onChange={setPage} disabled={listLoading} />
          </div>
          <div className="memory-detail" aria-label={t('记录详情')} aria-busy={detailLoading}>
            {detailError && <p className="memory-error" role="alert">{detailError}</p>}
            {detailData && !detailError ? <MemoryRecordDetail data={detailData} busy={detailLoading}
              evidencePage={evidencePage} historyPage={historyPage} sourcePage={sourcePage}
              setEvidencePage={setEvidencePage} setHistoryPage={setHistoryPage} setSourcePage={setSourcePage}
              sourceExpanded={sourceExpanded} setSourceExpanded={setSourceExpanded} />
              : !detailError && <p className="memory-empty">{t(selected ? '正在读取详情…' : '选择一条记录查看内容、依据和变更')}</p>}
          </div>
        </div>
      </section>
    </div>
    <footer className="memory-footer">{t('展示已提炼并保存的记录；新对话的提炼结果可能稍后出现')}</footer>
  </main>
}

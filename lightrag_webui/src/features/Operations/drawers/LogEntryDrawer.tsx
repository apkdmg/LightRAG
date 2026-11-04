import { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { CopyIcon } from 'lucide-react'
import { Kv } from '@/features/TokenInsights/ui'
import { LogEntry, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import { LevelChip } from '../components/LogRow'
import OpsLink from '../components/OpsLink'
import { RedactedText } from '../components/RedactedSpan'
import Skeleton from '../components/Skeleton'
import { useOpsNav } from '../context'
import { fmtCount, shortId, utcTitle } from '../format'
import { callSite, fmtLogTime, lineCount, logsView, pivotFilter } from '../logs'
import { useOpsPoll } from '../poll'
import Section from './Section'

const pivotClass = 'hover:bg-muted inline-flex min-h-8 items-center rounded-lg border px-2.5 text-[13px]'
const linkButton = 'font-mono text-xs underline underline-offset-2 break-all text-left'

async function copyText(text: string, ok: string, failed: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast.success(ok)
  } catch {
    toast.error(failed)
  }
}

/** The entry's context fields (FR-24): each ID drills to its inspector on the breadcrumb. */
function Context({ e }: { e: LogEntry }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const push = { push: true }
  const rows: [string, ReactNode][] = []
  if (e.workspace) {
    const ws = e.workspace
    rows.push([
      t('operations.logEntry.workspace'),
      <button key="ws" type="button" className="text-left break-all underline underline-offset-2" onClick={() => nav.openDrawer({ kind: 'workspace', id: ws }, push)}>
        {ws}
      </button>
    ])
  }
  if (e.request_id) {
    const req = e.request_id
    rows.push([
      t('operations.logEntry.request'),
      <span key="req" className="inline-flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-xs break-all">{req}</span>
        <button
          type="button"
          className="hover:bg-muted inline-flex size-6 items-center justify-center rounded"
          aria-label={t('operations.logEntry.copyRequest')}
          title={t('operations.logEntry.copyRequest')}
          onClick={() => copyText(req, t('operations.logEntry.requestCopied'), t('operations.logEntry.copyFailed'))}
        >
          <CopyIcon className="size-3.5" aria-hidden="true" />
        </button>
      </span>
    ])
  }
  if (e.client_request_id) rows.push([t('operations.logEntry.clientRequest'), <span key="creq" className="font-mono text-xs break-all">{e.client_request_id}</span>])
  if (e.run_id) {
    const run = e.run_id
    rows.push([
      t('operations.logEntry.run'),
      <button key="run" type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'run', id: run }, push)}>
        {run}
      </button>
    ])
  }
  if (e.track_id) rows.push([t('operations.logEntry.track'), <span key="trk" className="font-mono text-xs break-all">{e.track_id}</span>])
  if (e.doc_id) {
    const doc = e.doc_id
    const ws = e.workspace
    rows.push([
      t('operations.logEntry.doc'),
      ws ? (
        <button key="doc" type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'document', workspace: ws, id: doc }, push)}>
          {doc}
        </button>
      ) : (
        <span key="doc" className="font-mono text-xs break-all">
          {doc}
        </span>
      )
    ])
  }
  if (e.worker_id || e.pid !== null) {
    const wk = e.worker_id
    const label = e.pid !== null ? t('operations.runDrawer.pid', { pid: e.pid }) : shortId(wk)
    rows.push([
      t('operations.logEntry.process'),
      wk ? (
        <button key="wk" type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'worker', id: wk }, push)}>
          {label}
        </button>
      ) : (
        <span key="wk" className="font-mono text-xs">
          {label}
        </span>
      )
    ])
  }
  if (e.operation || e.source) rows.push([t('operations.logEntry.operation'), [e.operation, e.source].filter(Boolean).join(' · ')])
  if (e.fingerprint) {
    rows.push([
      t('operations.logEntry.fingerprint'),
      <span key="fp" className="font-mono text-xs">
        {e.fingerprint}
        {e.repeat_count > 1 ? ` · ${t('operations.logEntry.repeats', { count: fmtCount(e.repeat_count) })}` : ''}
      </span>
    ])
  }
  return (
    <div className="flex flex-col gap-1">
      {rows.map(([k, v]) => (
        <Kv key={k} k={k}>
          {v}
        </Kv>
      ))}
    </div>
  )
}

/** Pivots (FR-24): the Logs tab filtered on the same request, run, doc or call site. */
function Pivots({ e }: { e: LogEntry }) {
  const { t } = useTranslation()
  const links: { key: string; label: string; field: Parameters<typeof pivotFilter>[0]; value: string }[] = []
  if (e.request_id) links.push({ key: 'req', label: t('operations.logEntry.sameRequest'), field: 'request_id', value: e.request_id })
  if (e.run_id) links.push({ key: 'run', label: t('operations.logEntry.sameRun'), field: 'run_id', value: e.run_id })
  if (e.doc_id) links.push({ key: 'doc', label: t('operations.logEntry.sameDoc'), field: 'doc_id', value: e.doc_id })
  if (e.fingerprint) links.push({ key: 'fp', label: t('operations.logEntry.sameFingerprint'), field: 'fingerprint', value: e.fingerprint })
  if (!links.length) return null
  return (
    <div className="flex flex-wrap gap-2">
      {links.map((l) => (
        <OpsLink key={l.key} to={logsView({ ...pivotFilter(l.field, l.value), level: 'INFO' })} className={pivotClass}>
          {l.label}
        </OpsLink>
      ))}
    </div>
  )
}

/**
 * The Log entry drawer (FR-24, §2.9 flow 5): the full redacted message and
 * traceback, the context fields with drill-downs, pivots to the Logs tab and
 * "copy request ID". Everything was redacted when the line was written and
 * cannot be revealed. URL-addressable as `olog=<id>`.
 */
export default function LogEntryDrawer({ id }: { id: string }) {
  const { t } = useTranslation()
  const poll = useOpsPoll((signal) => opsApi.logEntry(id, signal), [id])
  const e = poll.data?.entry ?? null
  const missing = (poll.error?.kind === 'not_found' && !poll.data) || (!!poll.data && !e && poll.data.available !== false)
  return (
    <>
      {missing ? <p className="text-muted-foreground m-0 text-sm">{t('operations.logEntry.notFound')}</p> : null}
      {poll.data?.available === false ? <p className="text-muted-foreground m-0 text-sm">{t('operations.unavailableShort')}</p> : null}
      {poll.error && !poll.data && !missing ? <ErrorPanel error={poll.error} onRetry={poll.reload} /> : null}
      {!poll.data && !poll.error ? <Skeleton className="h-40 w-full" /> : null}
      {e ? (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <LevelChip level={e.level} />
            <span className="font-mono text-xs" title={e.logger || undefined}>
              {callSite(e)}
            </span>
            <span className="text-muted-foreground font-mono text-xs" title={utcTitle(e.ts)}>
              {fmtLogTime(e.ts)}
            </span>
            {e.logger && e.logger !== 'lightrag' ? <span className="text-muted-foreground font-mono text-xs">{e.logger}</span> : null}
          </div>
          <p className="bg-muted/40 m-0 rounded-lg border px-3 py-2 text-[13px] break-words whitespace-pre-wrap">
            <RedactedText text={e.msg} />
          </p>
          {e.has_exc ? (
            <Section title={t('operations.logEntry.traceback', { count: lineCount(e.exc) })}>
              {e.exc ? (
                <pre className="bg-muted/40 m-0 max-h-[50vh] overflow-auto rounded-lg border px-3 py-2 font-mono text-[11.5px] leading-relaxed break-words whitespace-pre-wrap">
                  <RedactedText text={e.exc} />
                </pre>
              ) : (
                <p className="text-muted-foreground m-0 text-xs">
                  {poll.data?.exc_withheld ? t('operations.logEntry.excWithheld') : t('operations.logEntry.excMissing', { type: e.exc_type ?? '' })}
                </p>
              )}
            </Section>
          ) : null}
          <Context e={e} />
          <Pivots e={e} />
        </div>
      ) : null}
      <p className="text-muted-foreground m-0 pb-2 text-xs">{t('operations.logEntry.maskNote')}</p>
    </>
  )
}

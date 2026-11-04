import { useTranslation } from 'react-i18next'
import { Kv } from '@/features/TokenInsights/ui'
import { LogEntry, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import LogSection from '../components/LogSection'
import { StaleChip } from '../components/Panel'
import Skeleton from '../components/Skeleton'
import StageTimeline from '../components/StageTimeline'
import { StatusChip } from '../components/StatusChip'
import { useOpsNav, useOpsShell } from '../context'
import { fmtClock, fmtCount, fmtLatency, shortId, utcTitle } from '../format'
import { useStale } from '../hooks'
import { normalizeLogEntry } from '../logsNormalize'
import { useOpsPoll } from '../poll'
import { docStatusState } from '../severity'
import { ErrorKindText } from './RunDrawer'
import Section from './Section'

/** The doc's log lines when the server sent them with the drawer (§3.7 `logs{count,first[≤50]}`); null otherwise. */
function drawerLogs(raw: unknown): { count: number | null; rows: LogEntry[] } | null {
  if (!raw || typeof raw !== 'object') return null
  const o = raw as { available?: unknown; count?: unknown; first?: unknown }
  if (o.available === false || !Array.isArray(o.first)) return null
  const rows = o.first.map(normalizeLogEntry).filter((e): e is LogEntry => e !== null)
  return { count: typeof o.count === 'number' ? o.count : null, rows }
}

const DETAIL_SECONDS = 15

/**
 * The Document drawer (§2.9, US-5), read-only: status (with the attempt
 * number of a failed doc), the stage timeline from the doc-status metadata
 * timings, the current classified failure (stage, kind, model, the run that
 * hit it), the redacted error, the failure history across attempts and up
 * to 50 LLM calls for this doc, and its WARNING+ log lines (from the
 * drawer's own answer when the server includes them, else one doc_id
 * query). Runs open the Run drawer on the breadcrumb.
 * The file name stays masked; revealing it is an audited action of a later
 * release.
 */
export default function DocumentDrawer({ workspace, docId }: { workspace: string; docId: string }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status, openInsights } = useOpsShell()
  const poll = useOpsPoll((signal) => opsApi.document(workspace, docId, signal), [workspace, docId], DETAIL_SECONDS)
  const { stale, ageSeconds } = useStale(poll.updatedAt, DETAIL_SECONDS)
  const data = poll.data
  const doc = data?.doc
  const calls = data?.calls ?? []
  const failures = data?.failures ?? []
  const failure = doc?.failure ?? null
  const logs = drawerLogs(data?.logs)
  const openRun = (runId: string) => nav.openDrawer({ kind: 'run', id: runId }, { push: true })
  const inStack = nav.stack.some((d) => d.kind === 'workspace' && d.id === workspace)
  return (
    <>
      <div className="flex flex-col gap-1">
        <p className="m-0 font-mono text-sm break-all">{doc?.file_path_masked || shortId(docId, 24)}</p>
        <p className="text-muted-foreground m-0 text-xs break-all">
          {t('operations.document.path', { workspace, id: shortId(docId, 16) })}
          {!inStack ? (
            <>
              {' · '}
              <button
                type="button"
                className="underline underline-offset-2"
                onClick={() => nav.openDrawer({ kind: 'workspace', id: workspace }, { push: true })}
              >
                {t('operations.document.openWorkspace')}
              </button>
            </>
          ) : null}
        </p>
        {stale && data ? <StaleChip ageSeconds={ageSeconds} /> : null}
      </div>
      {poll.error && !data ? (
        poll.error.kind === 'not_found' ? (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.document.notFound')}</p>
        ) : (
          <ErrorPanel error={poll.error} onRetry={poll.reload} />
        )
      ) : null}
      {poll.error && data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!data && !poll.error ? <Skeleton className="h-32 w-full" /> : null}
      {doc ? (
        <div className={stale ? 'ops-stale flex flex-col gap-4' : 'flex flex-col gap-4'}>
          <div className="flex flex-col gap-1">
            <Kv k={t('operations.document.status')}>
              <StatusChip state={doc.cancelled ? 'unknown' : docStatusState(doc.status)}>
                {doc.cancelled ? t('operations.docStatus.cancelled') : t(`operations.docStatus.${doc.status}`, { defaultValue: doc.status })}
              </StatusChip>
              {doc.status === 'failed' && typeof failure?.attempt === 'number' ? (
                <span className="text-muted-foreground ms-2 text-xs">{t('operations.document.attempt', { n: failure.attempt })}</span>
              ) : null}
            </Kv>
            {doc.track_id ? (
              <Kv k={t('operations.document.track')}>
                <span className="font-mono text-xs break-all">{doc.track_id}</span>
              </Kv>
            ) : null}
            <Kv k={t('operations.document.chunks')}>{fmtCount(doc.chunks_count ?? null)}</Kv>
            <Kv k={t('operations.document.created')}>
              <span title={utcTitle(doc.created_at)}>{fmtClock(doc.created_at ?? null)}</span>
            </Kv>
            <Kv k={t('operations.document.updated')}>
              <span title={utcTitle(doc.updated_at)}>{fmtClock(doc.updated_at ?? null)}</span>
            </Kv>
          </div>
          {failure && doc.status === 'failed' ? (
            <Section title={t('operations.document.currentFailure')}>
              <div className="flex flex-col gap-1">
                <Kv k={t('operations.document.failureKind')}>
                  <span className="font-medium">
                    <ErrorKindText kind={failure.error_kind} status={failure.http_status} />
                  </span>
                </Kv>
                {failure.stage ? <Kv k={t('operations.document.failureStage')}>{t(`operations.stage.${failure.stage}`, { defaultValue: failure.stage })}</Kv> : null}
                {failure.model ? (
                  <Kv k={t('operations.document.failureModel')}>
                    <span className="font-mono text-xs">{failure.model}</span>
                  </Kv>
                ) : null}
                {failure.first_failed_at ? (
                  <Kv k={t('operations.document.firstFailed')}>
                    <span title={utcTitle(failure.first_failed_at)}>{fmtClock(failure.first_failed_at)}</span>
                  </Kv>
                ) : null}
                {failure.last_run_id ? (
                  <Kv k={t('operations.document.lastRun')}>
                    <button type="button" className="font-mono text-xs underline underline-offset-2" onClick={() => openRun(failure.last_run_id as string)}>
                      {shortId(failure.last_run_id)}
                    </button>
                  </Kv>
                ) : null}
              </div>
            </Section>
          ) : null}
          <Section title={t('operations.document.timeline')}>
            <StageTimeline stages={doc.stages ?? []} />
          </Section>
          {doc.error_redacted ? (
            <Section title={t('operations.document.error')}>
              <p className="bg-muted/40 m-0 rounded-lg border px-2.5 py-2 font-mono text-xs break-words whitespace-pre-wrap">{doc.error_redacted}</p>
            </Section>
          ) : null}
          <Section title={t('operations.document.failures')}>
            {failures.length ? (
              <ul className="m-0 flex list-none flex-col gap-1.5 p-0 text-[13px]">
                {failures.map((f, i) => (
                  <li key={`${f.ts}-${i}`} className="flex flex-col gap-0.5">
                    <span className="flex flex-wrap items-center gap-x-2">
                      <span className="font-mono text-xs" title={utcTitle(f.ts)}>
                        {fmtClock(f.ts)}
                      </span>
                      {f.run_id ? (
                        <button type="button" className="font-mono text-xs underline underline-offset-2" onClick={() => openRun(f.run_id as string)}>
                          {t('operations.stuck.run', { id: shortId(f.run_id, 6) })}
                        </button>
                      ) : null}
                      {f.stage ? <span>{t(`operations.stage.${f.stage}`, { defaultValue: f.stage })}</span> : null}
                      <span className="font-medium">
                        <ErrorKindText kind={f.error_kind} status={f.http_status} />
                      </span>
                      {f.model ? <span className="font-mono text-xs">{f.model}</span> : null}
                      {typeof f.attempt === 'number' ? <span className="text-muted-foreground text-xs">{t('operations.document.attempt', { n: f.attempt })}</span> : null}
                    </span>
                    {f.message ? <span className="text-muted-foreground font-mono text-xs break-words">{f.message}</span> : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground m-0 text-xs">
                {status?.features?.runs === false ? t('operations.document.failuresOff') : t('operations.document.noFailures')}
              </p>
            )}
          </Section>
          <Section
            title={t('operations.document.calls')}
            aside={
              status?.features?.insights ? (
                <button type="button" className="text-xs underline underline-offset-2" onClick={() => openInsights(workspace)}>
                  {t('operations.document.explore')}
                </button>
              ) : null
            }
          >
            {calls.length ? (
              <div className="overflow-x-auto">
                <table className="ops-stack w-full min-w-[520px] text-[13px]">
                  <thead>
                    <tr className="text-muted-foreground border-b text-xs">
                      {(['time', 'purpose', 'model', 'tokens', 'latency', 'status'] as const).map((c) => (
                        <th key={c} className={`p-1.5 font-medium ${c === 'tokens' || c === 'latency' ? 'text-right' : 'text-left'}`}>
                          {t(`operations.document.call.${c}`)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {calls.map((c, i) => (
                      <tr key={`${c.ts}-${i}`} className="border-b last:border-0">
                        <td className="p-1.5 font-mono text-xs whitespace-nowrap" data-label={t('operations.document.call.time')} title={utcTitle(c.ts)}>
                          {fmtClock(c.ts)}
                        </td>
                        <td className="p-1.5" data-label={t('operations.document.call.purpose')}>
                          {c.purpose ? t(`insights.purpose.${c.purpose}`, { defaultValue: c.purpose }) : '–'}
                        </td>
                        <td className="p-1.5 font-mono text-xs" data-label={t('operations.document.call.model')}>
                          {c.model || '–'}
                          {c.billing === 'byo' ? ' (BYO)' : ''}
                        </td>
                        <td className="p-1.5 text-right tabular-nums" data-label={t('operations.document.call.tokens')}>
                          {fmtCount(c.prompt_tokens ?? null)} → {fmtCount(c.completion_tokens ?? null)}
                        </td>
                        <td className="p-1.5 text-right tabular-nums" data-label={t('operations.document.call.latency')}>
                          {fmtLatency(c.latency_ms ?? null)}
                        </td>
                        <td className="p-1.5" data-label={t('operations.document.call.status')}>
                          {c.cache_hit ? t('operations.document.cacheHit') : t(`operations.document.callStatus.${c.status}`, { defaultValue: c.status ?? '–' })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-muted-foreground m-0 text-xs">
                {data?.calls_available === false || status?.features?.insights === false ? t('operations.document.callsOff') : t('operations.document.noCalls')}
              </p>
            )}
            <p className="text-muted-foreground m-0 text-xs">{t('operations.document.callsNote')}</p>
          </Section>
          <LogSection
            title={typeof logs?.count === 'number' ? t('operations.document.logsCount', { count: logs.count }) : t('operations.document.logs')}
            filter={{ doc_id: docId, window: '7d' }}
            preloaded={logs?.rows}
          />
        </div>
      ) : null}
      <p className="text-muted-foreground m-0 pb-2 text-xs">{t('operations.maskNote')}</p>
    </>
  )
}

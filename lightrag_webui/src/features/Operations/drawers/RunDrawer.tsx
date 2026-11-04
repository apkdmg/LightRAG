import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Kv } from '@/features/TokenInsights/ui'
import { DocFailure, OpsRun, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import { StaleChip } from '../components/Panel'
import ProgressBar from '../components/ProgressBar'
import { RunStateLabel } from '../components/RunState'
import Skeleton from '../components/Skeleton'
import { ToneChip } from '../components/StatusChip'
import { useOpsNav, useOpsShell } from '../context'
import { fmtAge, fmtClock, fmtCount, shortId, utcTitle } from '../format'
import { useStale } from '../hooks'
import { useOpsPoll } from '../poll'
import { failureKindCounts, runDurationSeconds, runGuidance } from '../processing/runs'
import { isLiveRunState, isMultiWorker, runLookOf } from '../severity'
import Section from './Section'

/** A run that still holds its workspace refreshes with the Active list (5 s); a finished one at the drawer pace. */
const LIVE_SECONDS = 5
const DETAIL_SECONDS = 15

const linkButton = 'text-left underline underline-offset-2 break-all'

/** "provider budget (402)": the kind's name, with the HTTP status when the name does not already carry it. */
export function ErrorKindText({ kind, status }: { kind: string | null | undefined; status?: number | null }) {
  const { t } = useTranslation()
  if (!kind) return <>–</>
  const name = t(`operations.errorKind.${kind}`, { defaultValue: kind })
  return <>{typeof status === 'number' && !name.includes(String(status)) ? `${name} (${status})` : name}</>
}

/** The run's header facts (§2.9 Run drawer): state, where, what, when, progress and the owner process. */
export function RunSummary({ run, now, stale, multiWorker }: { run: OpsRun; now: number | null; stale?: boolean; multiWorker: boolean }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const duration = runDurationSeconds(run, now)
  const running = run.status === 'running'
  const guide = runGuidance(run.state, multiWorker)
  const look = runLookOf(run.state)
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col items-start gap-1">
        <span className="flex flex-wrap items-center gap-2">
          <RunStateLabel state={run.state} stale={stale} />
          {run.was_orphaned ? (
            <ToneChip tone="unknown" icon="dashed-circle">
              <span title={t('operations.runDrawer.wasOrphanedHint')}>{t('operations.runDrawer.wasOrphaned')}</span>
            </ToneChip>
          ) : null}
        </span>
        <p className="text-muted-foreground m-0 text-xs">{t(look.explain)}</p>
      </div>
      <div className="flex flex-col gap-1">
        <Kv k={t('operations.runDrawer.workspace')}>
          <button type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'workspace', id: run.workspace }, { push: true })}>
            {run.workspace}
          </button>
        </Kv>
        <Kv k={t('operations.runDrawer.kindTrigger')}>
          {run.kind ? t(`operations.kind.${run.kind}`, { defaultValue: run.kind }) : '–'}
          {' · '}
          {run.trigger ? t(`operations.trigger.${run.trigger}`, { defaultValue: run.trigger }) : '–'}
        </Kv>
        {run.label ? (
          <Kv k={t('operations.runDrawer.label')}>
            <span className="break-all">{run.label}</span>
          </Kv>
        ) : null}
        <Kv k={t('operations.runDrawer.started')}>
          <span title={utcTitle(run.started_at)}>{fmtClock(run.started_at ?? null)}</span>
        </Kv>
        {run.finished_at ? (
          <Kv k={t('operations.runDrawer.finished')}>
            <span title={utcTitle(run.finished_at)}>
              {fmtClock(run.finished_at)} · {t('operations.runDrawer.took', { age: fmtAge(duration) })}
            </span>
          </Kv>
        ) : (
          <Kv k={t('operations.runDrawer.elapsed')}>{fmtAge(duration)}</Kv>
        )}
        <Kv k={t('operations.runDrawer.progress')}>
          <span className="inline-flex flex-wrap items-center gap-2">
            <ProgressBar done={run.docs_done} total={run.docs_total} />
            {run.stage && running ? <span className="text-muted-foreground text-xs">{t(`operations.stage.${run.stage}`, { defaultValue: run.stage })}</span> : null}
          </span>
        </Kv>
        <Kv k={t('operations.runDrawer.failedDocs')}>{fmtCount(run.docs_failed ?? null)}</Kv>
        {running && typeof run.progress_age_s === 'number' ? (
          <Kv k={t('operations.runDrawer.lastProgress')}>{t('operations.runDrawer.ago', { age: fmtAge(run.progress_age_s) })}</Kv>
        ) : null}
        <Kv k={t('operations.runDrawer.process')}>
          <span className="inline-flex flex-wrap items-center gap-x-2">
            {run.worker_id ? (
              <button type="button" className={`${linkButton} font-mono text-xs`} onClick={() => nav.openDrawer({ kind: 'worker', id: run.worker_id as string }, { push: true })}>
                {typeof run.pid === 'number' ? t('operations.runDrawer.pid', { pid: run.pid }) : shortId(run.worker_id)}
              </button>
            ) : (
              <span className="font-mono text-xs">{typeof run.pid === 'number' ? t('operations.runDrawer.pid', { pid: run.pid }) : '–'}</span>
            )}
            {running && typeof run.heartbeat_age_s === 'number' ? (
              <span className="text-muted-foreground text-xs">{t('operations.runDrawer.heartbeat', { age: fmtAge(run.heartbeat_age_s) })}</span>
            ) : null}
            {run.worker_alive === false ? <span className="text-xs text-[color:var(--ops-crit)]">{t('operations.runDrawer.ownerGone')}</span> : null}
          </span>
        </Kv>
        {run.error_kind || run.error_summary ? (
          <Kv k={t('operations.runDrawer.error')}>
            <span className="flex flex-col">
              <span className="font-medium">
                <ErrorKindText kind={run.error_kind} />
              </span>
              {run.error_summary ? <span className="text-muted-foreground font-mono text-xs break-words">{run.error_summary}</span> : null}
            </span>
          </Kv>
        ) : null}
        {run.request_id ? (
          <Kv k={t('operations.runDrawer.request')}>
            <span className="font-mono text-xs break-all">{run.request_id}</span>
          </Kv>
        ) : null}
        <Kv k={t('operations.runDrawer.runId')}>
          <span className="font-mono text-xs break-all">{run.run_id}</span>
        </Kv>
      </div>
      {run.state === 'paused' && run.pause ? (
        <p className="m-0 text-xs">
          {typeof run.pause.status_code === 'number'
            ? t('operations.active.pausedCode', { code: run.pause.status_code, slot: run.pause.slot ?? '' })
            : t('operations.active.pausedNoCode')}
        </p>
      ) : null}
      {guide ? (
        <div className="flex flex-col gap-0.5 rounded-lg border border-dashed px-3 py-2 text-[13px]" role="note">
          <strong>{t('operations.runDrawer.whatToDo')}</strong>
          <span>{t(`operations.runDrawer.guide.${guide}`)}</span>
        </div>
      ) : null}
    </div>
  )
}

/** Documents that failed in this run, newest first; each opens its Document drawer. */
export function RunFailures({ failures, workspace }: { failures: DocFailure[]; workspace: string }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const counts = failureKindCounts(failures)
  return (
    <Section title={t('operations.runDrawer.failures', { count: failures.length })}>
      {failures.length ? (
        <>
          <p className="m-0 text-[13px]">
            {counts.map((c, i) => (
              <span key={c.key}>
                {i ? ' · ' : ''}
                <ErrorKindText kind={c.key} /> {fmtCount(c.count)}
              </span>
            ))}
          </p>
          <ul className="m-0 flex max-h-80 list-none flex-col overflow-y-auto p-0">
            {failures.map((f, i) => {
              const ws = f.workspace || workspace
              return (
                <li key={`${f.ts}-${f.doc_id ?? i}-${i}`}>
                  <button
                    type="button"
                    disabled={!f.doc_id}
                    className="hover:bg-muted grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-3 rounded-lg px-2 py-1.5 text-left text-[13px] disabled:hover:bg-transparent"
                    onClick={() => f.doc_id && nav.openDrawer({ kind: 'document', workspace: ws, id: f.doc_id }, { push: true })}
                  >
                    <span className="font-mono text-xs" title={utcTitle(f.ts)}>
                      {fmtClock(f.ts)}
                    </span>
                    <span className="truncate font-mono text-xs">{f.file_path_masked || shortId(f.doc_id ?? null, 16)}</span>
                    <span className="text-muted-foreground text-xs whitespace-nowrap">
                      {typeof f.attempt === 'number' ? t('operations.document.attempt', { n: f.attempt }) : ''}
                    </span>
                    <span className="col-span-3 text-xs">
                      {f.stage ? `${t(`operations.stage.${f.stage}`, { defaultValue: f.stage })} · ` : ''}
                      <span className="font-medium">
                        <ErrorKindText kind={f.error_kind} status={f.http_status} />
                      </span>
                      {f.model ? <span className="text-muted-foreground font-mono"> · {f.model}</span> : null}
                    </span>
                    {f.message ? <span className="text-muted-foreground col-span-3 truncate font-mono text-xs">{f.message}</span> : null}
                  </button>
                </li>
              )
            })}
          </ul>
        </>
      ) : (
        <p className="text-muted-foreground m-0 text-sm">{t('operations.runDrawer.noFailures')}</p>
      )}
    </Section>
  )
}

/**
 * The Run drawer (§2.9, flows 1 to 3), read-only: the run's state with its
 * plain-language explanation, where and what it is, its owner process (a
 * link to the Worker drawer), progress, outcome, "what to do" for a hung,
 * unresponsive, orphaned or paused run, and the documents that failed in it.
 * URL-addressable as `orun=<run_id>`; drilling to a workspace, worker or
 * document pushes onto the breadcrumb.
 */
export default function RunDrawer({ runId }: { runId: string }) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [fast, setFast] = useState(true)
  const poll = useOpsPoll((signal) => opsApi.run(runId, signal), [runId], fast ? LIVE_SECONDS : DETAIL_SECONDS)
  const run = poll.data?.run ?? null
  const live = isLiveRunState(run?.state)
  if (poll.data && live !== fast) setFast(live)
  const { stale, ageSeconds: age } = useStale(poll.updatedAt, fast ? LIVE_SECONDS : DETAIL_SECONDS)
  const unavailable = poll.data?.available === false
  const missing = (poll.error?.kind === 'not_found' && !poll.data) || (!!poll.data && !run && !unavailable)
  return (
    <>
      {stale && run ? <StaleChip ageSeconds={age} /> : null}
      {missing ? <p className="text-muted-foreground m-0 text-sm">{t('operations.runDrawer.notFound')}</p> : null}
      {unavailable ? <p className="text-muted-foreground m-0 text-sm">{t('operations.runDrawer.unavailable')}</p> : null}
      {poll.error && !poll.data && !missing ? <ErrorPanel error={poll.error} onRetry={poll.reload} /> : null}
      {poll.error && poll.data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!poll.data && !poll.error ? <Skeleton className="h-40 w-full" /> : null}
      {run ? (
        <div className={stale ? 'ops-stale flex flex-col gap-4' : 'flex flex-col gap-4'}>
          <RunSummary run={run} now={poll.updatedAt} stale={stale} multiWorker={isMultiWorker(status?.process.mode)} />
          <RunFailures failures={poll.data?.failures ?? []} workspace={run.workspace} />
          {typeof poll.data?.log_count === 'number' ? (
            <p className="text-muted-foreground m-0 text-xs">{t('operations.runDrawer.logCount', { count: poll.data.log_count })}</p>
          ) : null}
          <p className="text-muted-foreground m-0 text-xs">{t('operations.runDrawer.retention')}</p>
        </div>
      ) : null}
      <p className="text-muted-foreground m-0 pb-2 text-xs">{t('operations.maskNote')}</p>
    </>
  )
}

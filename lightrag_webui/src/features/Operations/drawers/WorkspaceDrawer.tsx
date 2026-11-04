import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { CopyIcon, FolderOpenIcon, PlayIcon } from 'lucide-react'
import { Kv, Seg } from '@/components/admin'
import { useActingWorkspace } from '@/hooks/useActingWorkspace'
import { useSettingsStore } from '@/stores/settings'
import { Feed, WorkspaceDetail, opsApi } from '../api'
import { OWNER_SUSPECT_STATES } from '../actions'
import ErrorPanel from '../components/ErrorPanel'
import LogSection from '../components/LogSection'
import MaskedValue from '../components/MaskedValue'
import OpsLink from '../components/OpsLink'
import { ApproxChip, Pager, StaleChip } from '../components/Panel'
import { RunStateLabel } from '../components/RunState'
import Skeleton from '../components/Skeleton'
import { StatusChip } from '../components/StatusChip'
import { WorkspaceActionBar } from '../components/WorkspaceActions'
import { useOpsNav, useOpsShell } from '../context'
import { ageSeconds, fmtAge, fmtClock, fmtCount, fmtStamp, shortId, utcTitle } from '../format'
import { viewForTarget } from '../hashState'
import { useStale } from '../hooks'
import { useOpsPoll } from '../poll'
import { DrainerText } from '../processing/EmailQueueView'
import { runDurationSeconds } from '../processing/runs'
import { docStatusState } from '../severity'
import { EMPTY_FEED, FeedState, mergeFeed } from './feed'
import { ErrorKindText } from './RunDrawer'
import Section from './Section'

/** §3.10: the drawer's feed refreshes every 3 s, the rest every 15 s. */
const DETAIL_SECONDS = 15
const FEED_SECONDS = 3
const DOC_STATUSES = ['all', 'failed', 'processing', 'pending', 'processed'] as const

function LiveState({ data, now, stale }: { data: WorkspaceDetail; now: number | null; stale: boolean }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const live = data.live
  const loaded = Array.isArray(data.loaded_in) ? data.loaded_in : null
  if (data.live_status === 'unresponsive') {
    return (
      <div className="flex flex-col items-start gap-1">
        <ApproxChip />
        <p className="text-muted-foreground m-0 text-sm">{t('operations.workspace.managerUnresponsive')}</p>
      </div>
    )
  }
  if (!live || !live.busy) {
    const idleText = !live && (!loaded || loaded.length === 0) ? t('operations.workspace.idleNotLoaded') : t('operations.workspace.idle')
    return (
      <div className="flex flex-col items-start gap-1">
        <StatusChip state="unknown">{idleText}</StatusChip>
        {live?.scanning ? <p className="text-muted-foreground m-0 text-xs">{t('operations.workspace.scanning')}</p> : null}
        {typeof live?.pending_enqueues === 'number' && live.pending_enqueues > 0 ? (
          <p className="text-muted-foreground m-0 text-xs">{t('operations.workspace.enqueues', { count: live.pending_enqueues })}</p>
        ) : null}
      </div>
    )
  }
  const paused = !!live.paused || !!live.provider_rejection
  // The run holding busy, from the run table: it says hung or orphaned where the live flags only say busy.
  const runs = data.recent_runs ?? []
  const current = (live.run_id ? runs.find((r) => r.run_id === live.run_id) : undefined) ?? runs.find((r) => r.status === 'running')
  const heldState = current && (current.status === 'running' || current.status === 'orphaned') ? current.state : 'running'
  const started = live.job_start ? ageSeconds(live.job_start, now ?? undefined) : null
  const notes: string[] = []
  if (typeof live.batchs === 'number' && live.batchs > 0) notes.push(t('operations.workspace.batch', { current: live.cur_batch ?? 0, total: live.batchs }))
  if (typeof live.docs === 'number') notes.push(t('operations.workspace.docs', { count: live.docs }))
  if (started !== null) notes.push(t('operations.workspace.runningFor', { age: fmtAge(started) }))
  return (
    <div className="flex flex-col items-start gap-1">
      <span className="flex flex-wrap items-center gap-2">
        <RunStateLabel state={paused ? 'paused' : heldState} stale={stale} />
        {live.job_name ? (
          <MaskedValue kind="job_name" workspace={data.workspace} id={live.run_id ?? null}>
            <span className="text-[13px] break-all">{live.job_name}</span>
          </MaskedValue>
        ) : null}
      </span>
      {notes.length ? <p className="text-muted-foreground m-0 text-xs">{notes.join(' · ')}</p> : null}
      {paused ? (
        <p className="m-0 text-xs">
          {typeof live.provider_rejection?.status_code === 'number'
            ? t('operations.active.pausedCode', { code: live.provider_rejection.status_code, slot: live.provider_rejection.slot ?? '' })
            : t('operations.active.pausedNoCode')}
        </p>
      ) : null}
      {live.run_id || current ? (
        <p className="m-0 text-xs">
          <button
            type="button"
            className="underline underline-offset-2"
            onClick={() => nav.openDrawer({ kind: 'run', id: (live.run_id ?? current?.run_id) as string }, { push: true })}
          >
            {t('operations.stuck.run', { id: shortId(live.run_id ?? current?.run_id) })}
          </button>
          {typeof (live.owner_pid ?? current?.pid) === 'number' ? (
            <span className="text-muted-foreground"> · {t('operations.runDrawer.pid', { pid: live.owner_pid ?? current?.pid })}</span>
          ) : null}
          {current?.worker_alive === false ? <span className="text-[color:var(--ops-crit)]"> · {t('operations.runDrawer.ownerGone')}</span> : null}
        </p>
      ) : null}
      {live.cancellation_requested ? <p className="m-0 text-xs">{t('operations.active.cancelling')}</p> : null}
      {live.destructive_busy ? <p className="m-0 text-xs">{t('operations.active.destructive')}</p> : null}
      {live.latest_message ? <p className="text-muted-foreground m-0 font-mono text-xs break-words">{live.latest_message}</p> : null}
    </div>
  )
}

/**
 * What the action row needs from the drawer's answer: the live flags, and
 * whether the owner of `busy` looks gone (its run is orphaned or
 * unresponsive, the OS confirmed the process exited, or busy has no run).
 */
function actionContext(d: WorkspaceDetail) {
  const live = d.live
  if (d.live_status === 'unresponsive') return { busy: null, emailPending: d.email_queue?.pending ?? 0 }
  const runs = d.recent_runs ?? []
  const current = (live?.run_id ? runs.find((r) => r.run_id === live.run_id) : undefined) ?? runs.find((r) => r.status === 'running' || r.status === 'orphaned')
  const ownerSuspect = !!live?.busy && (!current || OWNER_SUSPECT_STATES.has(current.state) || current.worker_alive === false)
  return {
    busy: live ? !!live.busy : false,
    destructiveBusy: live?.destructive_busy ?? null,
    cancellationRequested: live?.cancellation_requested ?? null,
    ownerSuspect,
    emailPending: d.email_queue?.pending ?? 0
  }
}

/** The workspace's `history_messages`, redacted on the server; appends every 3 s and replaces itself on a reset. */
function LiveFeed({ workspace }: { workspace: string }) {
  const { t } = useTranslation()
  const [feed, setFeed] = useState<FeedState>(EMPTY_FEED)
  const offsetRef = useRef<number | null>(null)
  const poll = useOpsPoll(
    async (signal) => {
      const requested = offsetRef.current
      const resp = await opsApi.feed(workspace, requested === null ? { limit: 200 } : { offset: requested, limit: 200 }, signal)
      return { resp, requested }
    },
    [workspace],
    FEED_SECONDS
  )
  // Merge each new answer once, during render (React's "state from the previous render" pattern).
  const [merged, setMerged] = useState<{ resp: Feed; requested: number | null } | null>(null)
  if (poll.data && poll.data !== merged) {
    setMerged(poll.data)
    setFeed((f) => mergeFeed(f, poll.data!.resp, poll.data!.requested))
  }
  useEffect(() => {
    // Nothing shown yet (or the feed was emptied): ask for the newest lines, not the oldest.
    offsetRef.current = feed.next > 0 ? feed.next : null
  }, [feed.next])

  const boxRef = useRef<HTMLDivElement | null>(null)
  const [follow, setFollow] = useState(true)
  useEffect(() => {
    const box = boxRef.current
    if (box && follow) box.scrollTop = box.scrollHeight
  }, [feed.lines, follow])
  const onScroll = () => {
    const box = boxRef.current
    if (!box) return
    const atBottom = Math.abs(box.scrollHeight - box.scrollTop - box.clientHeight) < 2
    if (atBottom !== follow) setFollow(atBottom)
  }

  return (
    <Section
      title={t('operations.workspace.feed')}
      aside={
        follow ? null : (
          <button type="button" className="hover:bg-muted inline-flex min-h-7 items-center gap-1 rounded-lg border px-2 text-xs" onClick={() => setFollow(true)}>
            <PlayIcon className="size-3" aria-hidden="true" />
            {t('operations.workspace.follow')}
          </button>
        )
      }
    >
      {poll.error && !poll.data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {feed.reset ? <p className="m-0 text-xs">{t('operations.workspace.feedReset')}</p> : null}
      <div
        ref={boxRef}
        onScroll={onScroll}
        className="bg-muted/40 max-h-64 min-h-16 overflow-y-auto rounded-lg border px-2.5 py-2 font-mono text-xs leading-relaxed"
        role="log"
        aria-label={t('operations.workspace.feed')}
      >
        {!poll.data && !poll.error ? (
          <Skeleton className="h-12 w-full" />
        ) : feed.lines.length ? (
          feed.lines.map((line, i) => (
            <div key={`${feed.next - feed.lines.length + i}`} className="break-words whitespace-pre-wrap">
              {line}
            </div>
          ))
        ) : (
          <span className="text-muted-foreground">{t('operations.workspace.feedEmpty')}</span>
        )}
      </div>
      <p className="text-muted-foreground m-0 text-xs">{t('operations.workspace.feedNote')}</p>
    </Section>
  )
}

function Documents({ workspace }: { workspace: string }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const [status, setStatus] = useState<(typeof DOC_STATUSES)[number]>('all')
  const [paging, setPaging] = useState({ status, page: 1 })
  const page = paging.status === status ? paging.page : 1
  const poll = useOpsPoll(
    (signal) => opsApi.documents(workspace, { status: status === 'all' ? undefined : status, page }, signal),
    [workspace, status, page],
    DETAIL_SECONDS
  )
  const data = poll.data
  return (
    <Section title={t('operations.workspace.documents')}>
      <Seg
        value={status}
        onChange={(v) => setStatus(v)}
        label={t('operations.workspace.documentsFilter')}
        options={DOC_STATUSES.map((s) => ({ value: s, label: t(s === 'all' ? 'operations.workspace.allDocs' : `operations.docStatus.${s}`) }))}
      />
      {poll.error && !data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!data && !poll.error ? <Skeleton className="h-20 w-full" /> : null}
      {data ? (
        data.items.length ? (
          <ul className={`m-0 flex list-none flex-col p-0 ${poll.loading ? 'opacity-60' : ''}`}>
            {data.items.map((d) => (
              <li key={d.doc_id}>
                <button
                  type="button"
                  className="hover:bg-muted grid w-full grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 rounded-lg px-2 py-1.5 text-left text-[13px]"
                  onClick={() => nav.openDrawer({ kind: 'document', workspace, id: d.doc_id }, { push: true })}
                >
                  <span className="truncate font-mono text-xs">{d.file_path_masked || shortId(d.doc_id, 16)}</span>
                  <StatusChip state={d.cancelled ? 'unknown' : docStatusState(d.status)}>
                    {d.cancelled ? t('operations.docStatus.cancelled') : t(`operations.docStatus.${d.status}`, { defaultValue: d.status })}
                  </StatusChip>
                  <span className="text-muted-foreground text-xs" title={utcTitle(d.updated_at)}>
                    {shortId(d.doc_id, 12)} · {fmtClock(d.updated_at ?? null)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.workspace.noDocs')}</p>
        )
      ) : null}
      {data ? <Pager page={data.page} pageSize={data.page_size} total={data.total} onPage={(p) => setPaging({ status, page: p })} /> : null}
      <p className="text-muted-foreground m-0 text-xs">{t('operations.maskNote')}</p>
    </Section>
  )
}

/**
 * The Workspace drawer (§2.9, US-4). Opening it never loads the workspace on
 * the server: counts, recent runs and failures by kind come from SQL and
 * live state from a read-only snapshot. When actions are on, the action row
 * offers Cancel, Release busy (multi-worker), Run pipeline now and Drain
 * email, each behind a confirm dialog with a reason. "Idle (not loaded in
 * memory)" when no process has it. Runs open the Run drawer on the
 * breadcrumb.
 */
export default function WorkspaceDrawer({ workspace }: { workspace: string }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status, openInsights } = useOpsShell()
  const acting = useActingWorkspace()
  const poll = useOpsPoll((signal) => opsApi.workspace(workspace, signal), [workspace], DETAIL_SECONDS)
  const { stale, ageSeconds: age } = useStale(poll.updatedAt, DETAIL_SECONDS)
  const d = poll.data
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(workspace)
      toast.success(t('operations.workspace.copied'))
    } catch {
      toast.error(t('operations.workspace.copyFailed'))
    }
  }
  /** OQ-13: Documents acts on this workspace (X-Target-Workspace on every call) under an "Acting on" banner. */
  const openInDocuments = () => {
    if (!acting.start(workspace)) {
      toast.error(t('operations.workspace.openDocumentsRefused'))
      return
    }
    useSettingsStore.getState().setCurrentTab('documents')
  }
  const c = d?.counts
  const eq = d?.email_queue
  const pauses = Array.isArray(d?.pause) ? d.pause : []
  return (
    <>
      <div className="flex flex-col gap-1">
        {d?.owner ? <p className="text-muted-foreground m-0 text-[13px] break-all">{t('operations.workspace.owner', { owner: d.owner })}</p> : null}
        {stale && d ? <StaleChip ageSeconds={age} /> : null}
      </div>
      {poll.error && !d ? (
        poll.error.kind === 'not_found' ? (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.workspace.notFound')}</p>
        ) : (
          <ErrorPanel error={poll.error} onRetry={poll.reload} />
        )
      ) : null}
      {poll.error && d ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!d && !poll.error ? <Skeleton className="h-28 w-full" /> : null}
      {d ? (
        <div className={stale ? 'ops-stale flex flex-col gap-4' : 'flex flex-col gap-4'}>
          <LiveState data={d} now={poll.updatedAt} stale={stale} />
          {d.action_in_progress?.actor ? (
            <p className="m-0 text-[13px]">
              {t('operations.workspace.actionInProgress', {
                actor: d.action_in_progress.actor,
                action: d.action_in_progress.action ?? '',
                time: fmtClock(d.action_in_progress.since ?? null)
              })}
            </p>
          ) : null}
          <WorkspaceActionBar
            workspace={workspace}
            ctx={actionContext(d)}
            facts={{ emailPending: d.email_queue?.pending ?? 0, paused: !!d.live?.paused || pauses.length > 0 }}
            onChanged={poll.reload}
          />
          {c ? (
            <Section title={t('operations.workspace.counts')}>
              <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
                {(['pending', 'parsing', 'analyzing', 'processing', 'processed', 'failed'] as const).map((k) => (
                  <Kv key={k} k={t(`operations.docStatus.${k}`)}>
                    {fmtCount(c[k] ?? null)}
                  </Kv>
                ))}
                <Kv k={t('operations.backlog.cancelled')}>{fmtCount(c.cancelled ?? null)}</Kv>
                <Kv k={t('operations.backlog.dup')}>{fmtCount(c.dup ?? null)}</Kv>
              </div>
            </Section>
          ) : null}
          {Array.isArray(d.loaded_in) && d.loaded_in.length ? (
            <Section title={t('operations.workspace.loadedIn')}>
              <p className="m-0 font-mono text-xs">{d.loaded_in.join(', ')}</p>
            </Section>
          ) : null}
          <Section title={t('operations.workspace.emailQueue')}>
            {eq && eq.pending > 0 ? (
              <div className="flex flex-col gap-1">
                <Kv k={t('operations.email.col.pending')}>{fmtCount(eq.pending)}</Kv>
                <Kv k={t('operations.email.col.since')}>
                  <span title={utcTitle(eq.pending_since)}>{fmtAge(eq.pending_since ? ageSeconds(eq.pending_since, poll.updatedAt ?? undefined) : null)}</span>
                </Kv>
                {eq.drainer || eq.backoff_until ? (
                  <Kv k={t('operations.email.col.drainer')}>
                    <DrainerText item={eq} now={poll.updatedAt} />
                  </Kv>
                ) : null}
                {typeof eq.attempts === 'number' ? <Kv k={t('operations.email.col.attempts')}>{fmtCount(eq.attempts)}</Kv> : null}
                {typeof eq.bad === 'number' && eq.bad > 0 ? <Kv k={t('operations.email.col.bad')}>{fmtCount(eq.bad)}</Kv> : null}
                {eq.last_result_class ? (
                  <Kv k={t('operations.email.col.last')}>
                    {t(`operations.email.result.${eq.last_result_class}`, { defaultValue: eq.last_result_class })}
                    {eq.reason ? ` · ${eq.reason}` : ''}
                  </Kv>
                ) : null}
              </div>
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{t('operations.workspace.emailEmpty')}</p>
            )}
          </Section>
          {pauses.length ? (
            <Section
              title={t('operations.workspace.pauses')}
              aside={
                <button type="button" className="text-xs underline underline-offset-2" onClick={() => useSettingsStore.getState().setCurrentTab('providers')}>
                  {t('operations.pauses.openProviders')}
                </button>
              }
            >
              <ul className="m-0 flex list-none flex-col gap-1 p-0 text-[13px]">
                {pauses.map((p, i) => (
                  <li key={`${p.slot}-${p.kind}-${i}`}>
                    <StatusChip state="paused">{t('operations.pauses.slotKind', { slot: p.slot ?? '–', kind: p.kind ?? '–' })}</StatusChip>
                    {typeof p.status_code === 'number' ? <span className="ms-2 font-mono text-xs">{p.status_code}</span> : null}
                    {p.byo ? <span className="ms-2 text-xs">{t('operations.pauses.byo')}</span> : null}
                    {p.since ? <span className="text-muted-foreground ms-2 text-xs">{t('operations.pauses.since', { time: fmtClock(p.since) })}</span> : null}
                  </li>
                ))}
              </ul>
            </Section>
          ) : null}
          {d.failures_by_kind?.length ? (
            <Section
              title={t('operations.workspace.failuresByKind')}
              aside={
                <OpsLink to={viewForTarget(nav.view, { tab: 'processing', sub: 'failures' })} className="text-xs underline underline-offset-2">
                  {t('operations.workspace.allFailures')}
                </OpsLink>
              }
            >
              <ul className="m-0 flex list-none flex-wrap gap-x-3 gap-y-1 p-0 text-[13px]">
                {d.failures_by_kind.map((f) => (
                  <li key={f.key}>
                    <ErrorKindText kind={f.key} status={f.http_status} /> <span className="font-mono text-xs">{fmtCount(f.count)}</span>
                  </li>
                ))}
              </ul>
              <p className="text-muted-foreground m-0 text-xs">{t('operations.workspace.failuresWindow')}</p>
            </Section>
          ) : null}
          <Section title={t('operations.workspace.recentRuns')}>
            {d.recent_runs?.length ? (
              <ul className="m-0 flex list-none flex-col p-0">
                {d.recent_runs.map((r) => (
                  <li key={r.run_id}>
                    <button
                      type="button"
                      className="hover:bg-muted flex w-full flex-wrap items-center gap-x-2.5 gap-y-0.5 rounded-lg px-2 py-1.5 text-left text-[13px]"
                      onClick={() => nav.openDrawer({ kind: 'run', id: r.run_id }, { push: true })}
                    >
                      <span className="font-mono text-xs" title={utcTitle(r.started_at)}>
                        {fmtStamp(r.started_at ?? null, poll.updatedAt ?? undefined)}
                      </span>
                      <span>{r.kind ? t(`operations.kind.${r.kind}`, { defaultValue: r.kind }) : '–'}</span>
                      {/* Recent runs are history: their dot never breathes (§2.12). */}
                      <RunStateLabel state={r.state} stale />
                      {typeof r.docs_total === 'number' ? (
                        <span className="font-mono text-xs">
                          {fmtCount(r.docs_done ?? 0)}/{fmtCount(r.docs_total)}
                        </span>
                      ) : null}
                      {typeof r.docs_failed === 'number' && r.docs_failed > 0 ? (
                        <span className="text-xs">{t('operations.active.failedDocs', { count: r.docs_failed })}</span>
                      ) : null}
                      <span className="text-muted-foreground text-xs">{fmtAge(runDurationSeconds(r, poll.updatedAt))}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground m-0 text-xs">
                {status?.features?.runs === false ? t('operations.workspace.runsOff') : t('operations.workspace.noRuns')}
              </p>
            )}
          </Section>
        </div>
      ) : null}
      <LiveFeed workspace={workspace} />
      <LogSection
        title={typeof d?.errors_1h === 'number' ? t('operations.workspace.logs', { count: d.errors_1h }) : t('operations.workspace.logsTitle')}
        filter={{ workspace, window: '24h' }}
      />
      <Documents workspace={workspace} />
      <div className="flex flex-wrap gap-2 pb-2">
        <button type="button" className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-3 text-[13px]" onClick={copy}>
          <CopyIcon className="size-3.5" aria-hidden="true" />
          {t('operations.workspace.copyId')}
        </button>
        <button
          type="button"
          className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-3 text-[13px]"
          onClick={openInDocuments}
          title={t('operations.workspace.openDocumentsHint')}
        >
          <FolderOpenIcon className="size-3.5" aria-hidden="true" />
          {acting.workspace === workspace ? t('operations.workspace.openDocumentsActive') : t('operations.workspace.openDocuments')}
        </button>
        {status?.features?.insights ? (
          <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px]" onClick={() => openInsights({ workspace })}>
            {t('operations.workspace.openInsights')}
          </button>
        ) : null}
      </div>
    </>
  )
}

import { useTranslation } from 'react-i18next'
import StatusCard from '@/components/status/StatusCard'
import { Kv } from '@/features/TokenInsights/ui'
import { OpsWorker, WorkerSample, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import { StaleChip } from '../components/Panel'
import Skeleton from '../components/Skeleton'
import Sparkline from '../components/Sparkline'
import { WorkerName } from '../components/WorkersTable'
import { useOpsShell } from '../context'
import { fmtAge, fmtBytes, fmtClock, fmtCount, fmtLatency, utcTitle } from '../format'
import { poolText, toLightragStatus, uptimeSeconds } from '../health/model'
import { WORKERS_SECONDS } from '../health/WorkersView'
import { useStale } from '../hooks'
import { useOpsPoll } from '../poll'
import { isMultiWorker } from '../severity'
import Section from './Section'

const HOURS = 24
/** Samples are written once a minute; the history refreshes at that pace. */
const HISTORY_SECONDS = 60

type Metric = { key: keyof Omit<WorkerSample, 't'>; format: (n: number) => string; color: string }

const METRICS: Metric[] = [
  { key: 'rss', format: fmtBytes, color: 'var(--ops-info)' },
  { key: 'cpu', format: (n) => `${Math.round(n)}%`, color: 'var(--ops-info)' },
  { key: 'loop_lag', format: fmtLatency, color: 'var(--ops-warn)' },
  { key: 'pool_wait', format: fmtLatency, color: 'var(--ops-warn)' }
]

/** Last, min and max of a series, for the text beside each sparkline. */
function stats(values: (number | null)[]) {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (!finite.length) return null
  return { last: finite[finite.length - 1], min: Math.min(...finite), max: Math.max(...finite) }
}

function History({ workerId }: { workerId: string }) {
  const { t } = useTranslation()
  const poll = useOpsPoll((signal) => opsApi.workerHistory(workerId, HOURS, signal), [workerId], HISTORY_SECONDS)
  const { stale } = useStale(poll.updatedAt, HISTORY_SECONDS)
  const samples = poll.data ?? []
  return (
    <Section title={t('operations.workerDrawer.history', { hours: HOURS })}>
      {poll.error && !poll.data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!poll.data && !poll.error ? <Skeleton className="h-40 w-full" /> : null}
      {poll.data ? (
        samples.length ? (
          <div className={stale ? 'ops-stale grid gap-3 sm:grid-cols-2' : 'grid gap-3 sm:grid-cols-2'}>
            {METRICS.map((m) => {
              const values = samples.map((s) => s[m.key])
              const s = stats(values)
              return (
                <div key={m.key} className="flex min-w-0 flex-col gap-1">
                  <span className="flex items-baseline justify-between gap-2 text-xs">
                    <span className="font-semibold">{t(`operations.workerDrawer.metric.${m.key}`)}</span>
                    <span className="text-muted-foreground tabular-nums">
                      {s ? t('operations.workerDrawer.stats', { last: m.format(s.last), min: m.format(s.min), max: m.format(s.max) }) : '–'}
                    </span>
                  </span>
                  <Sparkline values={values} color={m.color} />
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.workerDrawer.noHistory')}</p>
        )
      ) : null}
    </Section>
  )
}

/** The redacted config with this worker's queues, in the status dialog's own card (read-only). */
function ConfigAndQueues({ worker }: { worker: OpsWorker }) {
  const { t } = useTranslation()
  const config = useOpsPoll((signal) => opsApi.config(signal), [])
  const status = toLightragStatus(config.data, worker.snapshot)
  return (
    <Section title={t('operations.workerDrawer.config')}>
      {config.error && !config.data ? <ErrorPanel error={config.error} onRetry={config.reload} compact /> : null}
      {!config.data && !config.error ? <Skeleton className="h-24 w-full" /> : null}
      {config.data ? (
        status ? (
          <div className="overflow-x-auto">
            <StatusCard status={status} />
          </div>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.workerDrawer.configPartial')}</p>
        )
      ) : null}
      <p className="text-muted-foreground m-0 text-xs">{t('operations.workerDrawer.configNote')}</p>
    </Section>
  )
}

/**
 * The Worker drawer (§2.6): one process from its heartbeat row. Liveness,
 * memory, CPU, event-loop lag, the Postgres pool, queues, the Insights
 * recorder and log sink, email drainers and the leases it holds, then 24 h
 * sparklines and the config and queue card. Command line and environment are
 * never shown.
 */
export default function WorkerDrawer({ workerId }: { workerId: string }) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const poll = useOpsPoll((signal) => opsApi.workers(signal), [], WORKERS_SECONDS)
  const { stale, ageSeconds: age } = useStale(poll.updatedAt, WORKERS_SECONDS)
  const multi = isMultiWorker(status?.process.mode)
  const w = poll.data?.items.find((x) => x.worker_id === workerId) ?? null
  const s = w?.snapshot
  const now = poll.updatedAt ?? 0
  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        {w ? <WorkerName w={w} single={!multi} /> : null}
        {stale && w ? <StaleChip ageSeconds={age} /> : null}
      </div>
      {poll.error && !poll.data ? <ErrorPanel error={poll.error} onRetry={poll.reload} /> : null}
      {poll.error && poll.data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!poll.data && !poll.error ? <Skeleton className="h-40 w-full" /> : null}
      {poll.data && !w ? <p className="text-muted-foreground m-0 text-sm">{t('operations.workerDrawer.gone')}</p> : null}
      {w && s ? (
        <div className={stale ? 'ops-stale flex flex-col gap-4' : 'flex flex-col gap-4'}>
          <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
            <Kv k={t('operations.workerDrawer.mode')}>{w.mode ?? status?.process.mode ?? '–'}</Kv>
            <Kv k={t('operations.workerDrawer.started')}>
              <span title={utcTitle(w.started_at)}>{fmtClock(w.started_at)}</span>
            </Kv>
            <Kv k={t('operations.worker.col.up')}>{fmtAge(uptimeSeconds(w, now))}</Kv>
            <Kv k={t('operations.worker.col.heartbeat')}>{fmtAge(w.heartbeat_age_s)}</Kv>
            {w.exited_at ? (
              <Kv k={t('operations.workerDrawer.exited')}>
                <span title={utcTitle(w.exited_at)}>{fmtClock(w.exited_at)}</span>
              </Kv>
            ) : null}
            {w.version ? <Kv k={t('operations.system.version')}>{w.version}</Kv> : null}
            {w.host ? <Kv k={t('operations.workerDrawer.host')}>{w.host}</Kv> : null}
            <Kv k={t('operations.worker.col.cpu')}>{typeof s.cpu_pct === 'number' ? `${Math.round(s.cpu_pct)}%` : '–'}</Kv>
            <Kv k={t('operations.worker.col.rss')}>{fmtBytes(s.rss)}</Kv>
            <Kv k={t('operations.worker.col.fds')}>{fmtCount(s.fds)}</Kv>
            <Kv k={t('operations.workerDrawer.threads')}>{fmtCount(s.threads)}</Kv>
            <Kv k={t('operations.worker.col.instances')}>{fmtCount(s.instances)}</Kv>
            <Kv k={t('operations.worker.col.loopLag')}>{fmtLatency(s.loop_lag_ms)}</Kv>
            <Kv k={t('operations.worker.col.pool')}>{poolText(s.pool)}</Kv>
            {multi ? <Kv k={t('operations.worker.col.manager')}>{fmtLatency(s.manager_rtt_ms)}</Kv> : null}
          </div>
          <Section title={t('operations.workerDrawer.queues')}>
            <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-3">
              {(['llm', 'embed', 'rerank'] as const).map((k) => (
                <Kv key={k} k={t(`operations.worker.col.${k}`)}>
                  {s.queues[k] ? t('operations.workerDrawer.queue', { running: fmtCount(s.queues[k]!.running ?? null), queued: fmtCount(s.queues[k]!.queued ?? null) }) : '–'}
                </Kv>
              ))}
            </div>
          </Section>
          <Section title={t('operations.workerDrawer.recorder')}>
            {s.recorder ? (
              <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
                <Kv k={t('operations.workerDrawer.buffer')}>{fmtCount(s.recorder.buffer)}</Kv>
                <Kv k={t('operations.workerDrawer.lastFlush')}>
                  <span title={utcTitle(s.recorder.last_flush_ok_at)}>{fmtClock(s.recorder.last_flush_ok_at)}</span>
                </Kv>
                <Kv k={t('operations.workerDrawer.flushFailures')}>{fmtCount(s.recorder.flush_failures)}</Kv>
                <Kv k={t('operations.workerDrawer.lostRows')}>{fmtCount(s.recorder.lost_rows)}</Kv>
                <Kv k={t('operations.workerDrawer.droppedEvents')}>{fmtCount(s.recorder.dropped_events)}</Kv>
              </div>
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{t('operations.workerDrawer.recorderOff')}</p>
            )}
            {s.log_sink ? (
              <p className="text-muted-foreground m-0 text-xs">
                {t('operations.workerDrawer.logSink', { buffer: fmtCount(s.log_sink.buffer), dropped: fmtCount(s.log_sink.dropped), limited: fmtCount(s.log_sink.rate_limited) })}
              </p>
            ) : null}
          </Section>
          <Section title={t('operations.workerDrawer.drainers')}>
            {s.drainers.length ? (
              <ul className="m-0 flex list-none flex-col gap-0.5 p-0 text-[13px]">
                {s.drainers.map((d) => (
                  <li key={d.workspace} className="flex flex-wrap gap-2">
                    <span className="break-all">{d.workspace}</span>
                    {d.state ? <span className="text-muted-foreground text-xs">{t(`operations.workerDrawer.drainer.${d.state}`, { defaultValue: d.state })}</span> : null}
                    {d.backoff_until ? (
                      <span className="text-muted-foreground text-xs" title={utcTitle(d.backoff_until)}>
                        {t('operations.workerDrawer.backoff', { time: fmtClock(d.backoff_until) })}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{t('operations.workerDrawer.noDrainers')}</p>
            )}
          </Section>
          <Section title={t('operations.workerDrawer.leases')}>
            <p className="m-0 font-mono text-xs">{w.leases_held.length ? w.leases_held.join(', ') : t('operations.workerDrawer.noLeases')}</p>
          </Section>
        </div>
      ) : null}
      {w ? <History workerId={workerId} /> : null}
      {w ? <ConfigAndQueues worker={w} /> : null}
      <p className="text-muted-foreground m-0 pb-2 text-xs">{t('operations.workerDrawer.privacy')}</p>
    </>
  )
}

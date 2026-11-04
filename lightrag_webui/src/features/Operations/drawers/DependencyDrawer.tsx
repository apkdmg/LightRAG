import { useTranslation } from 'react-i18next'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { RefreshCwIcon, SendIcon } from 'lucide-react'
import { Kv } from '@/components/admin'
import { DepDetail, DepHealth, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import OpsLink from '../components/OpsLink'
import { StaleChip } from '../components/Panel'
import Skeleton from '../components/Skeleton'
import { StatusChip } from '../components/StatusChip'
import { useOpsNav } from '../context'
import { fmtAge, fmtBytes, fmtClock, fmtCount, fmtLatency, utcTitle } from '../format'
import { viewForTarget } from '../hashState'
import { useHealthChanged, useProbeActions } from '../health/actions'
import { GPU_DEP, normalizeGpus } from '../health/gpu'
import { GpuCards } from '../performance/GpuView'
import { PROBE_INTERVAL_SECONDS, canRecheck, depState, detailParts, drawerStatus } from '../health/model'
import { useStale } from '../hooks'
import { OPS_CHART } from '../motion'
import { useOpsPoll } from '../poll'
import Section from './Section'

const HOURS = 24
const axisStyle = { fontSize: 11, fill: 'var(--muted-foreground)' }

/** Plain-language notes for dependencies whose numbers are easy to misread (§2.9 flow 4, §2.13). */
const NOTES: Record<string, string> = {
  vllm_embed: 'operations.depDrawer.note.vllm',
  vllm_rerank: 'operations.depDrawer.note.vllm',
  embed: 'operations.depDrawer.note.vllm',
  rerank: 'operations.depDrawer.note.vllm',
  milvus: 'operations.depDrawer.note.milvus',
  vector: 'operations.depDrawer.note.vector',
  llm: 'operations.depDrawer.note.llm',
  smtp: 'operations.depDrawer.note.smtp',
  postgres: 'operations.depDrawer.note.postgres'
}

/** Where the dependency's own traffic is charted (Performance). */
const PIVOTS: Record<string, { tab: string; sub: string }> = {
  vllm_embed: { tab: 'performance', sub: 'deps' },
  vllm_rerank: { tab: 'performance', sub: 'deps' },
  embed: { tab: 'performance', sub: 'deps' },
  rerank: { tab: 'performance', sub: 'deps' },
  llm: { tab: 'performance', sub: 'llm' }
}

/** The window ends when the data was confirmed, so the axis does not drift between polls. */
function LatencyChart({ series, end }: { series: DepDetail['latency_series']; end: number }) {
  const { t } = useTranslation()
  const points = series.map((p) => ({ t: p.t, latency: p.latency_ms, down: p.status === 'down' ? 0 : null }))
  return (
    <div className="ops-dim h-[150px]" role="img" aria-label={t('operations.depDrawer.chartLabel', { hours: HOURS })}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={[end - HOURS * 3600_000, end]}
            tickFormatter={(v: number) => fmtClock(v).slice(0, 5)}
            tick={axisStyle}
            tickLine={false}
            axisLine={false}
          />
          <YAxis tick={axisStyle} tickLine={false} axisLine={false} width={48} tickFormatter={(v: number) => fmtLatency(v)} />
          <Tooltip
            isAnimationActive={false}
            labelFormatter={(v) => fmtClock(Number(v))}
            formatter={(value, name) => [name === 'down' ? t('operations.probe.down') : fmtLatency(Number(value)), t(`operations.depDrawer.series.${String(name)}`)]}
          />
          <Line dataKey="latency" stroke="var(--ops-info)" dot={false} strokeWidth={1.5} connectNulls={false} {...OPS_CHART} />
          <Line dataKey="down" stroke="var(--ops-crit)" strokeWidth={0} dot={{ r: 2.5, fill: 'var(--ops-crit)' }} {...OPS_CHART} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** The whitelisted detail keys with plain values, as key/value rows. */
function DetailRows({ detail }: { detail: Record<string, unknown> }) {
  const { t } = useTranslation()
  // The target has its own row above; nested lists (blocking, targets, paths) have their own sections.
  const rows = Object.entries(detail).filter(([k, v]) => k !== 'target' && (v === null || ['string', 'number', 'boolean'].includes(typeof v)))
  if (!rows.length) return null
  const fmt = (k: string, v: unknown) => {
    if (v === null || v === '') return '–'
    if (typeof v === 'boolean') return v ? t('operations.depDrawer.yes') : t('operations.depDrawer.no')
    if (typeof v === 'number') return k.endsWith('_ms') ? fmtLatency(v) : fmtCount(v)
    if (k.endsWith('_at')) return fmtClock(v as string)
    return String(v)
  }
  return (
    <div className="flex flex-col gap-1">
      {rows.map(([k, v]) => (
        <Kv key={k} k={t(`operations.depDrawer.key.${k}`, { defaultValue: k.replace(/_/g, ' ') })}>
          <span className="font-mono font-normal break-all">{fmt(k, v)}</span>
        </Kv>
      ))}
    </div>
  )
}

/** TLS targets and disk paths: one line each (`detail.targets`, `detail.paths`). */
function DetailLists({ detail }: { detail: Record<string, unknown> }) {
  const { t } = useTranslation()
  const rows = (v: unknown) => (Array.isArray(v) ? v.filter((r): r is Record<string, any> => !!r && typeof r === 'object') : [])
  const targets = rows(detail.targets)
  const paths = rows(detail.paths)
  if (!targets.length && !paths.length) return null
  return (
    <>
      {targets.length ? (
        <Section title={t('operations.depDrawer.targets')}>
          <ul className="m-0 flex list-none flex-col gap-0.5 p-0 font-mono text-xs">
            {targets.map((r, i) => (
              <li key={`${r.target}-${i}`}>
                {typeof r.days_left === 'number'
                  ? t('operations.depDrawer.targetRow', { target: r.target ?? '–', days: Math.floor(r.days_left) })
                  : t('operations.depDrawer.targetError', { target: r.target ?? '–', error: r.error_class ?? t('operations.probe.unknown') })}
                {r.self_signed ? ` · ${t('operations.depDrawer.targetSelfSigned')}` : ''}
                {r.trusted === false && !r.self_signed ? ` · ${t('operations.depDrawer.targetUntrusted')}` : ''}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
      {paths.length ? (
        <Section title={t('operations.depDrawer.paths')}>
          <ul className="m-0 flex list-none flex-col gap-0.5 p-0 font-mono text-xs">
            {paths.map((r, i) => (
              <li key={`${r.path}-${i}`}>
                {t('operations.depDrawer.pathRow', {
                  path: r.path ?? '–',
                  free: typeof r.free_pct === 'number' ? Math.round(r.free_pct) : '–',
                  used: fmtBytes(typeof r.used === 'number' ? r.used : null),
                  total: fmtBytes(typeof r.total === 'number' ? r.total : null)
                })}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </>
  )
}

function Blocking({ rows }: { rows: DepDetail['blocking'] }) {
  const { t } = useTranslation()
  const col = (k: string) => t(`operations.depDrawer.blocking.${k}`)
  return (
    <Section title={t('operations.depDrawer.blocking.title')}>
      {rows.length ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[460px] text-xs">
            <thead>
              <tr className="text-muted-foreground border-b">
                <th className="p-1.5 text-left font-medium">{col('pid')}</th>
                <th className="p-1.5 text-left font-medium">{col('blockedBy')}</th>
                <th className="p-1.5 text-right font-medium">{col('duration')}</th>
                <th className="p-1.5 text-left font-medium">{col('statement')}</th>
                <th className="p-1.5 text-left font-medium">{col('fingerprint')}</th>
                <th className="p-1.5 text-right font-medium">{col('length')}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((b, i) => (
                <tr key={`${b.pid}-${i}`} className="border-b last:border-0">
                  <td className="p-1.5 font-mono">{b.pid ?? '–'}</td>
                  <td className="p-1.5 font-mono">{b.blocked_by.length ? b.blocked_by.join(', ') : '–'}</td>
                  <td className="p-1.5 text-right tabular-nums">{fmtAge(b.duration_s ?? null)}</td>
                  <td className="p-1.5">
                    {b.statement_type ?? '–'}
                    {b.state ? <span className="text-muted-foreground"> · {b.state}</span> : null}
                    {b.wait_event ? <span className="text-muted-foreground"> · {b.wait_event}</span> : null}
                  </td>
                  <td className="p-1.5 font-mono">{b.fingerprint ?? '–'}</td>
                  <td className="p-1.5 text-right tabular-nums">{fmtCount(b.query_length ?? null)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-muted-foreground m-0 text-sm">{t('operations.depDrawer.blocking.none')}</p>
      )}
      <p className="text-muted-foreground m-0 text-xs">{t('operations.depDrawer.blocking.note')}</p>
    </Section>
  )
}

/**
 * The Dependency drawer (§2.6, §2.9): the current state with what it means,
 * a 24 h latency chart, the last 50 results, error classes and the
 * dependency's safe detail (Postgres: blocked and blocking backends with
 * durations, statement type and a query fingerprint, never query text).
 * Re-check runs the probe now (rate-limited on the server); for SMTP the
 * test connects, says EHLO and NOOP, and never sends mail.
 */
export default function DependencyDrawer({ dep }: { dep: string }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const poll = useOpsPoll((signal) => opsApi.dependency(dep, HOURS, signal), [dep], PROBE_INTERVAL_SECONDS)
  const { stale, ageSeconds: age } = useStale(poll.updatedAt, PROBE_INTERVAL_SECONDS)
  const { busy, recheck, smtp, smtpTest } = useProbeActions()
  useHealthChanged(poll.reload)
  const d = poll.data
  const card: DepHealth = d?.dep ?? { id: dep, status: drawerStatus(d), detail: d?.detail ?? {}, strip: [] }
  const status = drawerStatus(d)
  const parts = detailParts({ ...card, detail: { ...card.detail, ...(d?.detail ?? {}) } }).map(([k, p]) => t(k, p))
  const pivot = PIVOTS[dep]
  const isPostgres = dep === 'postgres'
  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        {d ? (
          <StatusChip state={depState(status)}>
            {status === 'unprobed' ? t('operations.topology.notProbed') : t(`operations.probe.${status}`)}
          </StatusChip>
        ) : null}
        {d && typeof card.latency_ms === 'number' ? <span className="font-mono text-[13px]">{fmtLatency(card.latency_ms)}</span> : null}
        {stale && d ? <StaleChip ageSeconds={age} /> : null}
      </div>
      {poll.error && !d ? (
        poll.error.kind === 'not_found' ? (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.depDrawer.notFound')}</p>
        ) : (
          <ErrorPanel error={poll.error} onRetry={poll.reload} />
        )
      ) : null}
      {poll.error && d ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!d && !poll.error ? <Skeleton className="h-40 w-full" /> : null}
      {d ? (
        <div className={stale ? 'ops-stale flex flex-col gap-4' : 'flex flex-col gap-4'}>
          <div className="flex flex-col gap-1">
            {card.target ? (
              <Kv k={t('operations.depDrawer.target')}>
                <span className="font-mono font-normal break-all">{card.target}</span>
              </Kv>
            ) : null}
            {card.checked_at ? (
              <Kv k={t('operations.depDrawer.checked')}>
                <span title={utcTitle(card.checked_at)}>{fmtClock(card.checked_at)}</span>
              </Kv>
            ) : null}
            {card.last_ok_at && status !== 'ok' ? (
              <Kv k={t('operations.health.lastOk')}>
                <span title={utcTitle(card.last_ok_at)}>{fmtClock(card.last_ok_at)}</span>
              </Kv>
            ) : null}
            <Kv k={t('operations.health.failures')}>{fmtCount(card.consecutive_failures ?? 0)}</Kv>
            <Kv k={t('operations.depDrawer.errorClass')}>
              <span className="font-mono">{card.error_class ?? '–'}</span>
            </Kv>
            {parts.length ? <p className="text-muted-foreground m-0 font-mono text-xs">{parts.join(' · ')}</p> : null}
          </div>
          {NOTES[dep] ? (
            <p className="m-0 rounded-lg border px-3 py-2 text-[13px]">{t(NOTES[dep])}</p>
          ) : null}
          <div className="flex flex-wrap gap-2">
            {canRecheck({ ...card, status }) ? (
              <button
                type="button"
                disabled={busy.has(dep)}
                aria-busy={busy.has(dep) || undefined}
                onClick={async () => {
                  if (await recheck(card)) poll.reload()
                }}
                className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-3 text-[13px] disabled:opacity-60"
                title={t('operations.health.recheckHint')}
              >
                <RefreshCwIcon className="size-3.5" aria-hidden="true" />
                {busy.has(dep) ? t('operations.health.working') : t('operations.depDrawer.recheck')}
              </button>
            ) : null}
            {dep === 'smtp' ? (
              <button
                type="button"
                disabled={smtp.pending}
                aria-busy={smtp.pending || undefined}
                onClick={() => void smtpTest()}
                className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-3 text-[13px] disabled:opacity-60"
                title={t('operations.health.smtpTestHint')}
              >
                <SendIcon className="size-3.5" aria-hidden="true" />
                {smtp.pending ? t('operations.health.working') : t('operations.health.smtpTest')}
              </button>
            ) : null}
            {pivot ? (
              <OpsLink to={viewForTarget(nav.view, pivot)} className="hover:bg-muted inline-flex min-h-8 items-center rounded-lg border px-3 text-[13px] no-underline">
                {t(`operations.depDrawer.pivot.${pivot.sub}`)}
              </OpsLink>
            ) : null}
          </div>
          {smtp.result || smtp.error ? (
            <p className="m-0 text-[13px]">
              {smtp.result
                ? smtp.result.ok
                  ? t('operations.health.smtpOk', { latency: fmtLatency(smtp.result.latency_ms ?? null) })
                  : t('operations.health.smtpFailed', { error: smtp.result.error_class ?? t('operations.state.unknown') })
                : t(smtp.error!.messageKey)}
            </p>
          ) : null}
          <Section title={t('operations.depDrawer.latency', { hours: HOURS })}>
            {d.latency_series.length ? <LatencyChart series={d.latency_series} end={poll.updatedAt ?? d.latency_series[d.latency_series.length - 1].t} /> : <p className="text-muted-foreground m-0 text-sm">{t('operations.depDrawer.noSeries')}</p>}
          </Section>
          {isPostgres ? <Blocking rows={d.blocking} /> : null}
          {dep === GPU_DEP ? (
            <Section
              title={t('operations.gpu.now')}
              aside={
                <OpsLink to={viewForTarget(nav.view, { tab: 'performance', sub: 'gpu' })} className="text-xs underline underline-offset-2">
                  {t('operations.gpu.openHistory')}
                </OpsLink>
              }
            >
              <GpuCards gpus={normalizeGpus({ ...card.detail, ...d.detail })} />
            </Section>
          ) : null}
          <DetailLists detail={{ ...card.detail, ...d.detail }} />
          {d.error_classes.length ? (
            <Section title={t('operations.depDrawer.errorClasses')}>
              <p className="m-0 font-mono text-xs">{d.error_classes.map((e) => `${e.key} ${fmtCount(e.count)}`).join(' · ')}</p>
            </Section>
          ) : null}
          <Section title={t('operations.depDrawer.detail')}>
            <DetailRows detail={{ ...card.detail, ...d.detail }} />
          </Section>
          <Section title={t('operations.depDrawer.recent', { count: d.recent.length })}>
            {d.recent.length ? (
              <ul className="m-0 flex max-h-72 list-none flex-col gap-0.5 overflow-y-auto p-0 font-mono text-xs">
                {d.recent.map((r, i) => (
                  <li key={`${r.ts}-${i}`} className="flex flex-wrap gap-x-2">
                    <span title={utcTitle(r.ts)}>{fmtClock(r.ts)}</span>
                    <span style={{ color: `var(--ops-${r.status === 'down' ? 'crit' : r.status === 'warn' ? 'warn' : r.status === 'ok' ? 'ok' : 'unknown'})` }}>
                      {t(`operations.probe.${r.status}`)}
                    </span>
                    <span>{fmtLatency(r.latency_ms ?? null)}</span>
                    {r.error_class ? <span>{r.error_class}</span> : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{t('operations.depDrawer.noRecent')}</p>
            )}
          </Section>
        </div>
      ) : null}
    </>
  )
}

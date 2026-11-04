import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import { Panel } from '../components/Panel'
import { useOpsShell } from '../context'
import { fmtCount, fmtLatency, fmtPercent, fmtRatePct } from '../format'
import { useOpsPoll } from '../poll'
import { ChartBlock, MarkerList, SeriesTable, TimeLines } from './charts'
import { PerfCtx, SeriesDef, latencySeries, windowRange } from './chartKit'
import { DEP_ROLES, DepPoint, DepRole, PerfDeps, VllmNow, latencyRows, perMinRows, perfFreshnessSeconds, perfRefreshSeconds } from './model'
import { ManualRefresh, MetricsOff, Th } from './shared'

/** Only the hourly rollup's average and max came back (no histogram percentiles). */
const isRollupOnly = (points: DepPoint[]) => points.length > 0 && points.every((p) => p.p95 === null) && points.some((p) => p.avg !== null)

/** "vLLM now: 3 running · 0 waiting · KV cache 22%" from the probe's /metrics read. */
function VllmLine({ v }: { v: VllmNow | null }) {
  const { t } = useTranslation()
  if (!v) return <span>{t('operations.perf.deps.vllmUnknown')}</span>
  return (
    <span>
      {t('operations.perf.deps.vllmNow', { running: fmtCount(v.running), waiting: fmtCount(v.waiting), kv: fmtPercent(v.kv_cache) })}
    </span>
  )
}

function RoleCharts({ role, data, from, to, ctx }: { role: DepRole; data: PerfDeps; from: number; to: number; ctx: PerfCtx }) {
  const { t } = useTranslation()
  const points = data.series[role]
  // Buckets with vLLM probe data but no calls carry no latency: left out of the latency chart.
  const called = points.filter((p) => p.calls !== 0)
  const rollup = isRollupOnly(called)
  const lat = latencySeries((k) => t(k), rollup)
  const rows = latencyRows(called, rollup)
  const name = t(`operations.perf.deps.role.${role}`)
  const rateSeries: SeriesDef[] = [{ key: 'rate', label: t('operations.perf.callsPerMin'), color: 'var(--ops-s1)' }]
  const rateRows = perMinRows(points, data.bucket_s)
  const vllmSeries: SeriesDef[] = [
    { key: 'running', label: t('operations.perf.running'), color: 'var(--ops-s1)' },
    { key: 'waiting', label: t('operations.perf.deps.waiting'), color: 'var(--ops-s2)' }
  ]
  const vllmRows = points.filter((p) => p.running !== null || p.waiting !== null).map((p) => ({ t: p.t, running: p.running, waiting: p.waiting }))
  const kvSeries: SeriesDef[] = [{ key: 'kv_cache', label: t('operations.perf.deps.kvCache'), color: 'var(--ops-s3)' }]
  const kvRows = points.filter((p) => p.kv_cache !== null).map((p) => ({ t: p.t, kv_cache: p.kv_cache }))
  const syncId = `ops-dep-${role}`
  return (
    <section className="flex min-w-0 flex-col gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="m-0 text-[14px] font-semibold">{name}</h3>
        <span className="text-muted-foreground text-xs">
          <VllmLine v={data.vllm[role]} />
        </span>
      </div>
      <ChartBlock
        title={t('operations.perf.deps.latency', { name })}
        series={lat}
        empty={rows.length ? undefined : t('operations.perf.deps.noCalls', { name })}
        chart={<TimeLines rows={rows} series={lat} from={from} to={to} format={fmtLatency} label={t('operations.perf.deps.latency', { name })} events={ctx.events} syncId={syncId} />}
        table={<SeriesTable rows={rows} series={lat} format={fmtLatency} caption={t('operations.perf.deps.latency', { name })} />}
      />
      {rateRows.length ? (
        <ChartBlock
          title={t('operations.perf.deps.rate', { name })}
          series={rateSeries}
          chart={<TimeLines rows={rateRows} series={rateSeries} from={from} to={to} format={(v) => fmtCount(v)} label={t('operations.perf.deps.rate', { name })} events={ctx.events} height={110} syncId={syncId} />}
          table={<SeriesTable rows={rateRows} series={rateSeries} format={(v) => fmtCount(v)} caption={t('operations.perf.deps.rate', { name })} />}
        />
      ) : null}
      {vllmRows.length ? (
        <ChartBlock
          title={t('operations.perf.deps.vllmTitle')}
          sub={t('operations.perf.deps.vllmNote')}
          series={vllmSeries}
          chart={<TimeLines rows={vllmRows} series={vllmSeries} from={from} to={to} format={(v) => fmtCount(v)} label={t('operations.perf.deps.vllmTitle')} events={ctx.events} height={110} syncId={syncId} />}
          table={<SeriesTable rows={vllmRows} series={vllmSeries} format={(v) => fmtCount(v)} caption={t('operations.perf.deps.vllmTitle')} />}
        />
      ) : null}
      {kvRows.length ? (
        <ChartBlock
          title={t('operations.perf.deps.kvTitle')}
          series={kvSeries}
          chart={<TimeLines rows={kvRows} series={kvSeries} from={from} to={to} format={(v) => fmtPercent(v)} label={t('operations.perf.deps.kvTitle')} events={ctx.events} height={90} syncId={syncId} />}
          table={<SeriesTable rows={kvRows} series={kvSeries} format={(v) => fmtPercent(v)} caption={t('operations.perf.deps.kvTitle')} />}
        />
      ) : null}
    </section>
  )
}

function SummaryTable({ data }: { data: PerfDeps }) {
  const { t } = useTranslation()
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[640px] text-[13px]">
        <caption className="sr-only">{t('operations.perf.deps.tableCaption')}</caption>
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <Th>{t('operations.perf.deps.roleCol')}</Th>
            <Th right>{t('operations.perf.calls')}</Th>
            <Th right>p50</Th>
            <Th right>p95</Th>
            <Th right>p99</Th>
            <Th right>{t('operations.perf.errorPct')}</Th>
            <Th right>{t('operations.perf.deps.vllmCol')}</Th>
          </tr>
        </thead>
        <tbody>
          {DEP_ROLES.map((role) => {
            const s = data.summary[role]
            const v = data.vllm[role]
            return (
              <tr key={role} className="border-b last:border-0">
                <td className="p-1.5" data-label={t('operations.perf.deps.roleCol')}>
                  {t(`operations.perf.deps.role.${role}`)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.calls')}>
                  {fmtCount(s.calls)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label="p50">
                  {fmtLatency(s.p50)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label="p95">
                  {fmtLatency(s.p95)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label="p99">
                  {fmtLatency(s.p99)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.errorPct')}>
                  {fmtRatePct(s.error_rate)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.deps.vllmCol')}>
                  {v ? `${fmtCount(v.running)} / ${fmtCount(v.waiting)} · ${fmtPercent(v.kv_cache)}` : '–'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/**
 * Performance › Embed & Rerank (§2.7, FR-41): latency percentiles, call
 * rate and errors from the timing wrapper inside the queue wrapper, with
 * vLLM's own running, waiting and KV cache from the probe. vLLM "waiting"
 * includes callers that reach it through the proxy, so the call rate
 * charted here is LightRAG's own share.
 */
export default function DepLatencyView({ ctx }: { ctx: PerfCtx }) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const poll = useOpsPoll((signal) => opsApi.perfDeps(ctx.window, signal), [ctx.window], perfRefreshSeconds(ctx.window))
  const off = status?.features?.metrics === false || poll.data?.enabled === false
  return (
    <Panel
      poll={poll}
      freshness={perfFreshnessSeconds(ctx.window)}
      title={t('operations.perf.deps.title')}
      sub={t('operations.perf.deps.sub')}
      actions={<ManualRefresh window={ctx.window} poll={poll} />}
    >
      {(data) => {
        const { from, to } = windowRange(ctx.window, data.generated_at, poll.updatedAt)
        return (
          <div className="flex flex-col gap-4">
            {off ? <MetricsOff /> : null}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {DEP_ROLES.map((role) => (
                <RoleCharts key={role} role={role} data={data} from={from} to={to} ctx={ctx} />
              ))}
            </div>
            <SummaryTable data={data} />
            <MarkerList events={ctx.events} from={from} to={to} />
          </div>
        )
      }}
    </Panel>
  )
}

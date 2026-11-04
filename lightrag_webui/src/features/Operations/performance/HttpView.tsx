import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import OpsLink from '../components/OpsLink'
import { Panel } from '../components/Panel'
import { useOpsShell } from '../context'
import { fmtCount, fmtLatency, fmtRatePct } from '../format'
import { useOpsPoll } from '../poll'
import { ChartBlock, MarkerList, SeriesTable, TimeLines } from './charts'
import { PerfCtx, Q_COLORS, SeriesDef, windowRange } from './chartKit'
import { HttpRoute, PerfWindow, errorShareRows, perMinRows, perfFreshnessSeconds, perfRefreshSeconds, routeLogsView, topRoutes } from './model'
import { ManualRefresh, MetricsOff, Th } from './shared'

function RoutesTable({ routes, window }: { routes: HttpRoute[]; window: PerfWindow }) {
  const { t } = useTranslation()
  const [all, setAll] = useState(false)
  const top = topRoutes(routes)
  const shown = all ? [...routes].sort((a, b) => (b.p95 ?? -1) - (a.p95 ?? -1) || a.key.localeCompare(b.key)) : top
  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto">
        <table className="ops-stack w-full min-w-[680px] text-[13px]">
          <caption className="sr-only">{t('operations.perf.http.tableCaption')}</caption>
          <thead>
            <tr className="text-muted-foreground border-b text-xs">
              <Th>{t('operations.perf.http.route')}</Th>
              <Th right>{t('operations.perf.http.perMin')}</Th>
              <Th right>p50</Th>
              <Th right>p95</Th>
              <Th right>p99</Th>
              <Th right>{t('operations.perf.http.err5xx')}</Th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.key} className="border-b last:border-0">
                <td className="p-1.5" data-label={t('operations.perf.http.route')}>
                  <OpsLink
                    to={routeLogsView(r.route, window)}
                    className="font-mono text-xs break-all underline-offset-2 hover:underline"
                    title={t('operations.perf.http.openLogs')}
                  >
                    {r.method ? <span className="text-muted-foreground me-1.5">{r.method}</span> : null}
                    {r.unmatched ? t('operations.perf.http.unmatched') : r.route}
                  </OpsLink>
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.http.perMin')}>
                  {r.rps === null ? '–' : fmtCount(r.rps * 60)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label="p50">
                  {fmtLatency(r.p50)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label="p95">
                  {fmtLatency(r.p95)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label="p99">
                  {fmtLatency(r.p99)}
                </td>
                <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.http.err5xx')}>
                  {fmtRatePct(r.err5xx)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!routes.length ? <p className="text-muted-foreground m-0 p-2 text-[13px]">{t('operations.perf.http.noRequests')}</p> : null}
      </div>
      <div className="text-muted-foreground flex flex-wrap items-center justify-between gap-2 text-xs">
        <span>{all ? t('operations.perf.http.allNote') : t('operations.perf.http.topNote', { shown: top.length, total: routes.length })}</span>
        {routes.length > top.length ? (
          <button type="button" aria-pressed={all} onClick={() => setAll((v) => !v)} className="hover:bg-muted min-h-7 rounded-md border px-2 text-xs">
            {all ? t('operations.perf.http.showTop') : t('operations.perf.http.showAll', { count: routes.length })}
          </button>
        ) : null}
      </div>
    </div>
  )
}

/**
 * Performance › HTTP (§2.7, FR-42): requests per minute, p95 and the 5xx
 * share over time from the timing step in RequestIdMiddleware, and the ten
 * slowest route templates by p95 with `<unmatched>` as one row. A route
 * opens Logs at level ≥ ERROR filtered to its path.
 */
export default function HttpView({ ctx }: { ctx: PerfCtx }) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const poll = useOpsPoll((signal) => opsApi.perfHttp(ctx.window, signal), [ctx.window], perfRefreshSeconds(ctx.window))
  const off = status?.features?.metrics === false || poll.data?.enabled === false
  return (
    <Panel
      poll={poll}
      freshness={perfFreshnessSeconds(ctx.window)}
      title={t('operations.perf.http.title')}
      sub={t('operations.perf.http.sub')}
      actions={<ManualRefresh window={ctx.window} poll={poll} />}
    >
      {(data) => {
        const { from, to } = windowRange(ctx.window, data.generated_at, poll.updatedAt)
        const rate: SeriesDef[] = [{ key: 'rate', label: t('operations.perf.http.perMin'), color: 'var(--ops-s1)' }]
        const lat: SeriesDef[] = [
          { key: 'p50', label: 'p50', color: Q_COLORS.p50, width: 1.5 },
          { key: 'p95', label: 'p95', color: Q_COLORS.p95 }
        ]
        const err: SeriesDef[] = [{ key: 'share', label: t('operations.perf.http.err5xx'), color: 'var(--ops-crit)' }]
        const rateRows = perMinRows(data.series, data.bucket_s)
        const latRows = data.series.map((p) => ({ t: p.t, p50: p.p50, p95: p.p95 }))
        const errRows = errorShareRows(data.series)
        const empty = data.series.length ? undefined : t('operations.perf.noPoints')
        return (
          <div className="flex flex-col gap-4">
            {off ? <MetricsOff /> : null}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <ChartBlock
                title={t('operations.perf.http.rateTitle')}
                series={rate}
                empty={empty}
                chart={<TimeLines rows={rateRows} series={rate} from={from} to={to} format={(v) => fmtCount(v)} label={t('operations.perf.http.rateTitle')} events={ctx.events} syncId="ops-http" height={150} />}
                table={<SeriesTable rows={rateRows} series={rate} format={(v) => fmtCount(v)} caption={t('operations.perf.http.rateTitle')} />}
              />
              <ChartBlock
                title={t('operations.perf.http.latencyTitle')}
                series={lat}
                empty={empty}
                chart={<TimeLines rows={latRows} series={lat} from={from} to={to} format={fmtLatency} label={t('operations.perf.http.latencyTitle')} events={ctx.events} syncId="ops-http" height={150} />}
                table={<SeriesTable rows={latRows} series={lat} format={fmtLatency} caption={t('operations.perf.http.latencyTitle')} />}
              />
              <ChartBlock
                title={t('operations.perf.http.errTitle')}
                series={err}
                empty={empty}
                chart={<TimeLines rows={errRows} series={err} from={from} to={to} format={fmtRatePct} label={t('operations.perf.http.errTitle')} events={ctx.events} syncId="ops-http" height={150} />}
                table={<SeriesTable rows={errRows} series={err} format={fmtRatePct} caption={t('operations.perf.http.errTitle')} />}
              />
            </div>
            {data.capped ? <p className="m-0 text-xs">{t('operations.perf.http.capped')}</p> : null}
            <RoutesTable routes={data.routes} window={ctx.window} />
            <MarkerList events={ctx.events} from={from} to={to} />
          </div>
        )
      }}
    </Panel>
  )
}

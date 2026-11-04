import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Seg } from '@/components/admin'
import { opsApi } from '../api'
import { Panel } from '../components/Panel'
import { useOpsShell } from '../context'
import { fmtCount, fmtLatency, fmtRatePct } from '../format'
import { useOpsPoll } from '../poll'
import { ChartBlock, MarkerList, SeriesTable, TimeLines } from './charts'
import { PerfCtx, SeriesDef, latencySeries, windowRange } from './chartKit'
import { LlmGroup, LlmRow, PerfLlm, latencyRows, perfFreshnessSeconds, perfRefreshSeconds, seriesFor, topKeys } from './model'
import { ManualRefresh, Th } from './shared'

/** Small multiples: one latency chart per purpose (or model), the busiest first. */
const MAX_CHARTS = 4

function useKeyLabel(group: LlmGroup) {
  const { t } = useTranslation()
  return (key: string) => (group === 'purpose' ? t(`insights.purpose.${key}`, { defaultValue: key }) : key || t('operations.perf.llm.unknownModel'))
}

function RatesTable({ data, group, insights }: { data: PerfLlm; group: LlmGroup; insights: boolean }) {
  const { t } = useTranslation()
  const { openInsights } = useOpsShell()
  const label = useKeyLabel(group)
  const rollup = data.approx_7d
  const rows = [...data.table].sort((a, b) => (b.calls ?? 0) - (a.calls ?? 0))
  const lat = (r: LlmRow) => (rollup ? [r.avg, r.max] : [r.p50, r.p95, r.p99])
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[720px] text-[13px]">
        <caption className="sr-only">{t('operations.perf.llm.tableCaption')}</caption>
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <Th>{t(`operations.perf.llm.group.${group}`)}</Th>
            <Th right>{t('operations.perf.calls')}</Th>
            {rollup ? (
              <>
                <Th right>{t('operations.perf.avg')}</Th>
                <Th right>{t('operations.perf.max')}</Th>
              </>
            ) : (
              <>
                <Th right>p50</Th>
                <Th right>p95</Th>
                <Th right>p99</Th>
              </>
            )}
            <Th right>{t('operations.perf.errorPct')}</Th>
            <Th right>{t('operations.perf.truncatedPct')}</Th>
            <Th right>{t('operations.perf.llm.queue')}</Th>
            <Th>{insights ? <span className="sr-only">{t('operations.perf.llm.explore')}</span> : null}</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-b last:border-0">
              <td className="p-1.5" data-label={t(`operations.perf.llm.group.${group}`)}>
                <span className={group === 'model' ? 'font-mono text-xs break-all' : undefined}>{label(r.key)}</span>
              </td>
              <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.calls')}>
                {fmtCount(r.calls)}
              </td>
              {lat(r).map((v, i) => (
                <td key={i} className="p-1.5 text-right font-mono text-xs" data-label={rollup ? t(i ? 'operations.perf.max' : 'operations.perf.avg') : ['p50', 'p95', 'p99'][i]}>
                  {fmtLatency(v)}
                </td>
              ))}
              <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.errorPct')}>
                {fmtRatePct(r.error_rate)}
              </td>
              <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.truncatedPct')}>
                {fmtRatePct(r.truncated_rate)}
              </td>
              <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.llm.queue')}>
                {r.running === null && r.queued === null ? '–' : `${fmtCount(r.running)}/${fmtCount(r.queued)}`}
              </td>
              <td className="p-1.5 text-right" data-label="">
                {insights ? (
                  <button
                    type="button"
                    className="text-xs whitespace-nowrap underline underline-offset-2"
                    onClick={() => openInsights(group === 'purpose' ? { tab: 'explore', purpose: r.key } : { tab: 'explore' })}
                  >
                    {t('operations.perf.llm.explore')}
                  </button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length ? <p className="text-muted-foreground m-0 p-2 text-[13px]">{t('operations.perf.llm.noCalls')}</p> : null}
    </div>
  )
}

/** Running and queued LLM calls, summed over every loaded instance: "is it us or the provider?" */
function QueueBlock({ data, from, to, ctx }: { data: PerfLlm; from: number; to: number; ctx: PerfCtx }) {
  const { t } = useTranslation()
  const series: SeriesDef[] = [
    { key: 'running', label: t('operations.perf.running'), color: 'var(--ops-s1)' },
    { key: 'queued', label: t('operations.perf.queued'), color: 'var(--ops-s2)' }
  ]
  const rows = data.queue_series.map((p) => ({ t: p.t, running: p.running, queued: p.queued }))
  const now = data.queue
  const nowText = now
    ? t('operations.perf.llm.queueNow', { running: fmtCount(now.running), queued: fmtCount(now.queued) })
    : t('operations.perf.llm.queueUnknown')
  if (!rows.length) return <p className="m-0 text-[13px]">{nowText}</p>
  return (
    <ChartBlock
      title={t('operations.perf.llm.queueTitle')}
      sub={nowText}
      series={series}
      chart={<TimeLines rows={rows} series={series} from={from} to={to} format={(v) => fmtCount(v)} label={t('operations.perf.llm.queueTitle')} events={ctx.events} height={120} syncId="ops-llm" />}
      table={<SeriesTable rows={rows} series={series} format={(v) => fmtCount(v)} caption={t('operations.perf.llm.queueTitle')} />}
    />
  )
}

/**
 * Performance › LLM (§2.7, FR-40): p50/p95/p99 per purpose or model from the
 * usage table for windows up to 24 h; the 7-day window reads hourly rollups
 * and says so. Below, the summed running and queued counts across every
 * loaded instance separate "the provider is slow" from "our queues are
 * full", and the rates table links each row to Insights Explore.
 */
export default function LlmView({ ctx }: { ctx: PerfCtx }) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [group, setGroup] = useState<LlmGroup>('purpose')
  const poll = useOpsPoll((signal) => opsApi.perfLlm(ctx.window, group, signal), [ctx.window, group], perfRefreshSeconds(ctx.window))
  const label = useKeyLabel(group)
  return (
    <Panel
      poll={poll}
      freshness={perfFreshnessSeconds(ctx.window)}
      title={t('operations.perf.llm.title')}
      sub={ctx.window === '7d' ? t('operations.perf.llm.rollupNote') : t('operations.perf.llm.liveNote')}
      actions={
        <>
          <Seg
            value={group}
            onChange={setGroup}
            label={t('operations.perf.llm.groupBy')}
            options={[
              { value: 'purpose' as LlmGroup, label: t('operations.perf.llm.group.purpose') },
              { value: 'model' as LlmGroup, label: t('operations.perf.llm.group.model') }
            ]}
          />
          <ManualRefresh window={ctx.window} poll={poll} />
        </>
      }
    >
      {(data) => {
        const { from, to } = windowRange(ctx.window, data.generated_at, poll.updatedAt)
        const rollup = data.approx_7d
        const series = latencySeries((k) => t(k), rollup)
        const keys = topKeys(data, MAX_CHARTS)
        const others = new Set([...data.table.map((r) => r.key), ...data.series.map((p) => p.key)]).size - keys.length
        if (!data.insights) {
          // No recorder, no latency source: the queue counts still come from heartbeats.
          return (
            <div className="flex flex-col gap-4">
              <p className="m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]">{t('operations.perf.llm.insightsOff')}</p>
              <QueueBlock data={data} from={from} to={to} ctx={ctx} />
            </div>
          )
        }
        return (
          <div className="flex flex-col gap-4">
            {keys.length ? (
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {keys.map((key) => {
                  const rows = latencyRows(seriesFor(data.series, key), rollup)
                  const title = t('operations.perf.llm.chartTitle', { name: label(key) })
                  return (
                    <ChartBlock
                      key={key}
                      title={title}
                      series={series}
                      empty={rows.length ? undefined : t('operations.perf.noPoints')}
                      chart={<TimeLines rows={rows} series={series} from={from} to={to} format={fmtLatency} label={title} events={ctx.events} syncId="ops-llm" />}
                      table={<SeriesTable rows={rows} series={series} format={fmtLatency} caption={title} />}
                    />
                  )
                })}
              </div>
            ) : (
              <p className="text-muted-foreground m-0 text-[13px]">{t('operations.perf.llm.noCalls')}</p>
            )}
            {others > 0 ? <p className="text-muted-foreground m-0 text-xs">{t('operations.perf.llm.moreKeys', { count: others })}</p> : null}
            <QueueBlock data={data} from={from} to={to} ctx={ctx} />
            <RatesTable data={data} group={group} insights={!!status?.features?.insights} />
            <MarkerList events={ctx.events} from={from} to={to} />
          </div>
        )
      }}
    </Panel>
  )
}

import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import { Panel } from '../components/Panel'
import { useOpsShell } from '../context'
import { fmtCount } from '../format'
import { useOpsPoll } from '../poll'
import { ChartBlock, MarkerList, SeriesTable, TimeBars, TimeLines } from './charts'
import { PerfCtx, SeriesDef, windowRange } from './chartKit'
import { perfFreshnessSeconds, perfRefreshSeconds, throughputDays } from './model'
import { ManualRefresh } from './shared'

const DAY_MS = 24 * 3600_000

/**
 * Performance › Throughput (§2.7, FR-43): documents processed and failed per
 * hour (from runs), LLM calls per minute (≤ 24 h) and email batches per
 * hour. Throughput reads whole days (1, or 7 for the 7 d window), so a short
 * window shows the last day. Longer history is the Insights Work tab's.
 */
export default function ThroughputView({ ctx }: { ctx: PerfCtx }) {
  const { t } = useTranslation()
  const { status, openInsights } = useOpsShell()
  const days = throughputDays(ctx.window)
  const poll = useOpsPoll((signal) => opsApi.perfThroughput(days, signal), [days], perfRefreshSeconds(ctx.window))
  return (
    <Panel
      poll={poll}
      freshness={perfFreshnessSeconds(ctx.window)}
      title={t('operations.perf.throughput.title')}
      sub={t('operations.perf.throughput.sub', { count: days })}
      actions={
        <>
          {status?.features?.insights ? (
            <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-2.5 text-[13px]" onClick={() => openInsights({ tab: 'work' })}>
              {t('operations.perf.throughput.work')}
            </button>
          ) : null}
          <ManualRefresh window={ctx.window} poll={poll} />
        </>
      }
    >
      {(data) => {
        const to = windowRange(ctx.window, data.generated_at, poll.updatedAt).to
        const from = to - data.days * DAY_MS
        const docs: SeriesDef[] = [
          { key: 'processed', label: t('operations.perf.throughput.processed'), color: 'var(--ops-info)' },
          { key: 'failed', label: t('operations.perf.throughput.failed'), color: 'var(--ops-crit)' }
        ]
        const docRows = data.docs.map((p) => ({ t: p.t, processed: p.value, failed: p.failed }))
        const batches: SeriesDef[] = [{ key: 'batches', label: t('operations.perf.throughput.batches'), color: 'var(--ops-s1)' }]
        const batchRows = data.email_batches.map((p) => ({ t: p.t, batches: p.value }))
        const calls: SeriesDef[] = [{ key: 'calls', label: t('operations.perf.callsPerMin'), color: 'var(--ops-s1)' }]
        const callRows = data.llm_calls_per_min.map((p) => ({ t: p.t, calls: p.value }))
        const count = (v: number) => fmtCount(v)
        return (
          <div className="flex flex-col gap-4">
            <ChartBlock
              title={t('operations.perf.throughput.docsTitle')}
              series={docs}
              empty={docRows.length ? undefined : status?.features?.runs === false ? t('operations.perf.throughput.runsOff') : t('operations.perf.throughput.noDocs')}
              chart={<TimeBars rows={docRows} series={docs} from={from} to={to} format={count} label={t('operations.perf.throughput.docsTitle')} events={ctx.events} />}
              table={<SeriesTable rows={docRows} series={docs} format={count} caption={t('operations.perf.throughput.docsTitle')} />}
            />
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <ChartBlock
                title={t('operations.perf.throughput.callsTitle')}
                series={calls}
                empty={callRows.length ? undefined : days > 1 ? t('operations.perf.throughput.callsNeedDay') : t('operations.perf.noPoints')}
                chart={<TimeLines rows={callRows} series={calls} from={Math.max(from, to - DAY_MS)} to={to} format={count} label={t('operations.perf.throughput.callsTitle')} events={ctx.events} height={150} />}
                table={<SeriesTable rows={callRows} series={calls} format={count} caption={t('operations.perf.throughput.callsTitle')} />}
              />
              <ChartBlock
                title={t('operations.perf.throughput.batchesTitle')}
                series={batches}
                empty={batchRows.length ? undefined : t('operations.perf.throughput.noBatches')}
                chart={<TimeBars rows={batchRows} series={batches} from={from} to={to} format={count} label={t('operations.perf.throughput.batchesTitle')} events={ctx.events} height={150} />}
                table={<SeriesTable rows={batchRows} series={batches} format={count} caption={t('operations.perf.throughput.batchesTitle')} />}
              />
            </div>
            <MarkerList events={ctx.events} from={from} to={to} />
          </div>
        )
      }}
    </Panel>
  )
}

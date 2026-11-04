import { useTranslation } from 'react-i18next'
import { Meter } from '@/components/admin'
import { opsApi } from '../api'
import { Panel } from '../components/Panel'
import { StatusChip } from '../components/StatusChip'
import { fmtClock, utcTitle } from '../format'
import { GpuMetric, GpuReading, PerfGpu, gpuColor, gpuKey, gpuLine, gpuWarn } from '../health/gpu'
import { depState, toProbeStatus } from '../health/model'
import { useOpsPoll } from '../poll'
import { ChartBlock, MarkerList, SeriesTable, TimeLines } from './charts'
import { PerfCtx, SeriesDef, windowRange } from './chartKit'
import { perfFreshnessSeconds, perfRefreshSeconds } from './model'
import { ManualRefresh } from './shared'

/** One tile per GPU: utilisation and memory as bars with their numbers, temperature and power beside. */
export function GpuCards({ gpus }: { gpus: GpuReading[] }) {
  const { t } = useTranslation()
  if (!gpus.length) return <p className="text-muted-foreground m-0 text-sm">{t('operations.gpu.noGpus')}</p>
  return (
    <ul className="m-0 grid list-none grid-cols-1 gap-3 p-0 md:grid-cols-2">
      {gpus.map((g) => {
        const line = gpuLine(g)
        const hot = g.temp_c !== null && g.temp_c >= 85
        return (
          <li key={g.index} className="flex min-w-0 flex-col gap-2 rounded-lg border p-3" data-gpu={g.index}>
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-[13px] font-semibold">{t('operations.gpu.title', { index: g.index })}</span>
              <span className="text-muted-foreground truncate text-xs">{g.name ?? ''}</span>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">{t('operations.gpu.util')}</span>
                <span className="font-mono font-semibold">{line.util}</span>
              </div>
              <Meter value={g.util_pct ?? 0} color="var(--ops-s1)" />
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">{t('operations.gpu.memory')}</span>
                <span className="font-mono font-semibold">{line.mem}</span>
              </div>
              <Meter value={g.mem_pct ?? 0} color={g.mem_pct !== null && g.mem_pct >= 95 ? 'var(--ops-warn)' : 'var(--ops-s1)'} />
            </div>
            <dl className="m-0 grid grid-cols-2 gap-2 text-xs">
              <div>
                <dt className="text-muted-foreground">{t('operations.gpu.temp')}</dt>
                <dd className="m-0 font-mono font-semibold">{line.temp}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">{t('operations.gpu.power')}</dt>
                <dd className="m-0 font-mono font-semibold">{line.power}</dd>
              </div>
            </dl>
            {gpuWarn(g) ? (
              <StatusChip state="warning" className="self-start">
                {hot ? t('operations.gpu.hot') : t('operations.gpu.memoryFull')}
              </StatusChip>
            ) : null}
          </li>
        )
      })}
    </ul>
  )
}

const FORMAT: Record<GpuMetric, (v: number) => string> = {
  util: (v) => `${Math.round(v)}%`,
  mem: (v) => `${Math.round(v)}%`,
  temp: (v) => `${Math.round(v)} °C`,
  power: (v) => `${Math.round(v)} W`
}

/** The window's charts: one measure per chart, one line per GPU, sharing the hover (`syncId`). */
export function GpuCharts({ data, from, to, ctx }: { data: PerfGpu; from: number; to: number; ctx: PerfCtx }) {
  const { t } = useTranslation()
  const { gpus, rows } = data.history
  const series: SeriesDef[] = gpus.map((id, i) => ({ key: gpuKey(id), label: t('operations.gpu.title', { index: id }), color: gpuColor(i) }))
  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
      {(['util', 'mem', 'temp', 'power'] as GpuMetric[]).map((metric) => {
        const shown = rows[metric]
        const hasValues = shown.some((r) => gpus.some((id) => typeof r[gpuKey(id)] === 'number'))
        const title = t(`operations.gpu.chart.${metric}`)
        return (
          <ChartBlock
            key={metric}
            title={title}
            sub={t(`operations.gpu.chartSub.${metric}`)}
            series={series}
            empty={hasValues ? undefined : t('operations.gpu.noHistory')}
            chart={<TimeLines rows={shown} series={series} from={from} to={to} format={FORMAT[metric]} label={title} events={ctx.events} syncId="ops-gpu" height={160} />}
            table={<SeriesTable rows={shown} series={series} format={FORMAT[metric]} caption={title} />}
          />
        )
      })}
    </div>
  )
}

/**
 * Performance › GPU (D5, Phase 2): utilisation, memory, temperature and
 * power per GPU from the nvidia_gpu_exporter sidecar, which the probe cycle
 * reads every 30 s (OPS_GPU_EXPORTER_URL), with history for the window from
 * the probe rows. The API container never touches the GPU itself; vLLM's KV
 * cache and queues stay in Embed & Rerank. Charts never animate.
 */
export default function GpuView({ ctx }: { ctx: PerfCtx }) {
  const { t } = useTranslation()
  const poll = useOpsPoll((signal) => opsApi.perfGpu(ctx.window, signal), [ctx.window], perfRefreshSeconds(ctx.window))
  const off = poll.data?.enabled === false
  if (off) {
    return (
      <p className="text-muted-foreground m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]" role="note">
        {t('operations.gpu.off')}
      </p>
    )
  }
  const latest = poll.data?.latest ?? null
  const status = toProbeStatus(latest?.status)
  return (
    <div className="flex flex-col gap-4">
      <Panel
        poll={poll}
        freshness={perfFreshnessSeconds(ctx.window)}
        title={t('operations.gpu.now')}
        sub={latest?.checked_at ? <span title={utcTitle(latest.checked_at)}>{t('operations.gpu.checked', { time: fmtClock(latest.checked_at) })}</span> : t('operations.gpu.sub')}
        actions={
          <>
            <StatusChip state={latest ? depState(status) : 'unknown'} />
            <ManualRefresh window={ctx.window} poll={poll} />
          </>
        }
      >
        {(data) =>
          data.latest ? (
            <>
              {status === 'down' ? (
                <p className="m-0 text-[13px] text-[color:var(--ops-crit)]">{t('operations.gpu.exporterDown', { error: data.latest.error_class ?? t('operations.state.unknown') })}</p>
              ) : null}
              <GpuCards gpus={data.latest.gpus} />
            </>
          ) : (
            <p className="text-muted-foreground m-0 text-sm">{t('operations.gpu.noReading')}</p>
          )
        }
      </Panel>
      <Panel poll={poll} freshness={perfFreshnessSeconds(ctx.window)} title={t('operations.gpu.historyTitle')} sub={t('operations.gpu.historySub')}>
        {(data) => {
          const { from, to } = windowRange(ctx.window, data.generated_at, poll.updatedAt)
          return (
            <>
              <GpuCharts data={data} from={from} to={to} ctx={ctx} />
              <MarkerList events={ctx.events} from={from} to={to} />
            </>
          )
        }}
      </Panel>
      <p className="text-muted-foreground m-0 text-xs">{t('operations.gpu.note')}</p>
    </div>
  )
}

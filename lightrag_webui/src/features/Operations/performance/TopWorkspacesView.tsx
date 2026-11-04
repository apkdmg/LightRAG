import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Meter, Seg } from '@/components/admin'
import { opsApi } from '../api'
import OpsLink from '../components/OpsLink'
import { Panel } from '../components/Panel'
import { useOpsNav } from '../context'
import { fmtCount, fmtDuration } from '../format'
import { useOpsPoll } from '../poll'
import { TOP_METRICS, TopMetric, perfFreshnessSeconds, perfRefreshSeconds } from './model'
import { PerfCtx } from './chartKit'
import { ManualRefresh, Th } from './shared'

const formatValue = (metric: TopMetric, v: number) => (metric === 'processing_minutes' ? fmtDuration(v * 60_000) : fmtCount(v))

/**
 * Performance › Top workspaces (§2.7, FR-45): the 20 busiest workspaces in
 * the window by LLM calls, processing minutes or failures, ranked by the
 * server (`ORDER BY … LIMIT 20`, never a per-workspace loop). A row opens the
 * Workspace drawer.
 */
export default function TopWorkspacesView({ ctx }: { ctx: PerfCtx }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const [metric, setMetric] = useState<TopMetric>('llm_calls')
  const poll = useOpsPoll((signal) => opsApi.perfTopWorkspaces(metric, ctx.window, signal), [metric, ctx.window], perfRefreshSeconds(ctx.window))
  return (
    <Panel
      poll={poll}
      freshness={perfFreshnessSeconds(ctx.window)}
      title={t('operations.perf.top.title')}
      sub={t(`operations.perf.top.sub.${metric}`)}
      actions={
        <>
          <Seg value={metric} onChange={setMetric} label={t('operations.perf.top.metric')} options={TOP_METRICS.map((m) => ({ value: m, label: t(`operations.perf.top.metrics.${m}`) }))} />
          <ManualRefresh window={ctx.window} poll={poll} />
        </>
      }
    >
      {(data) => {
        const max = data.items[0]?.value || 1
        return data.items.length ? (
          <div className="overflow-x-auto">
            <table className="ops-stack w-full min-w-[520px] text-[13px]">
              <caption className="sr-only">{t('operations.perf.top.title')}</caption>
              <thead>
                <tr className="text-muted-foreground border-b text-xs">
                  <Th>#</Th>
                  <Th>{t('operations.perf.workspace')}</Th>
                  <Th>
                    <span className="sr-only">{t('operations.perf.top.share')}</span>
                  </Th>
                  <Th right>{t(`operations.perf.top.metrics.${data.metric}`)}</Th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((r, i) => (
                  <tr key={r.workspace} className="border-b last:border-0">
                    <td className="text-muted-foreground p-1.5 font-mono text-xs" data-label="#">
                      {i + 1}
                    </td>
                    <td className="p-1.5" data-label={t('operations.perf.workspace')}>
                      <OpsLink to={{ ...nav.view, drawer: { kind: 'workspace', id: r.workspace } }} className="break-all underline-offset-2 hover:underline">
                        {r.workspace}
                      </OpsLink>
                    </td>
                    <td className="w-[30%] p-1.5" data-label="">
                      <Meter value={(r.value / max) * 100} color="var(--ops-s1)" />
                    </td>
                    <td className="p-1.5 text-right font-mono text-xs" data-label={t(`operations.perf.top.metrics.${data.metric}`)}>
                      {formatValue(data.metric, r.value)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-muted-foreground m-0 text-[13px]">{t('operations.perf.top.none')}</p>
        )
      }}
    </Panel>
  )
}

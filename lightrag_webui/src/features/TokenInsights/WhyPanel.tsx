import { useEffect } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { XIcon, ChevronLeftIcon, ChevronRightIcon } from 'lucide-react'
import { insightsApi } from './api'
import { HoursChart, WhySteps } from './charts'
import { useData, useInsights } from './context'
import { PURPOSES } from './meta'
import { Empty, PurposeName, RowButton, fmtDay, fmtPct, fmtTokens } from './ui'

const shift = (day: string, by: number) => {
  const d = new Date(day + 'T12:00:00')
  d.setDate(d.getDate() + by)
  return d.toISOString().slice(0, 10)
}

export default function WhyPanel() {
  const { t } = useTranslation()
  const reduced = useReducedMotion()
  const { scope, set, goExplore } = useInsights()
  const day = scope.day as string
  const { data, error } = useData(() => insightsApi.day(day, { workspace: scope.workspace }), [day, scope.workspace])
  useEffect(() => {
    const close = (e: KeyboardEvent) => e.key === 'Escape' && set({ day: null })
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [set])
  const today = new Date().toISOString().slice(0, 10)
  const rows: any[] = data?.purposes ?? []
  const changed = rows.filter((r) => r.verdict !== 'normal' && r.verdict !== 'unknown').sort((a, b) => Math.abs(b.tokens - (b.usual || 0)) - Math.abs(a.tokens - (a.usual || 0)))
  const main = changed[0]
  return (
    <motion.aside
      role="dialog"
      aria-label={t('insights.why.title')}
      initial={reduced ? false : { x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={reduced ? undefined : { x: 40, opacity: 0 }}
      transition={{ duration: 0.24 }}
      className="bg-background absolute top-0 right-0 bottom-0 z-30 flex w-full max-w-[560px] flex-col gap-3 overflow-y-auto border-l p-5 shadow-xl"
    >
      <div className="flex items-start justify-between gap-2.5">
        <div>
          <div className="text-muted-foreground text-xs">{t('insights.why.title')}{scope.workspace ? ` · ${scope.workspace}` : ''}</div>
          <h2 className="m-0 text-xl font-semibold">{fmtDay(day, true)}{data?.partial ? ` · ${t('insights.chart.soFar')}` : ''}</h2>
        </div>
        <div className="flex gap-1">
          <button type="button" className="hover:bg-muted rounded-lg border p-2" aria-label={t('insights.why.previous')} onClick={() => set({ day: shift(day, -1) })}><ChevronLeftIcon className="size-4" /></button>
          <button type="button" disabled={day >= today} className="hover:bg-muted rounded-lg border p-2 disabled:opacity-40" aria-label={t('insights.why.next')} onClick={() => set({ day: shift(day, 1) })}><ChevronRightIcon className="size-4" /></button>
          <button type="button" className="hover:bg-muted rounded-lg border p-2" aria-label={t('insights.why.close')} onClick={() => set({ day: null })}><XIcon className="size-4" /></button>
        </div>
      </div>
      {error ? <Empty>{error}</Empty> : null}
      {!data ? <Empty>{t('insights.loading')}</Empty> : (
        <>
          <p className="m-0 text-[15px] font-medium">
            {!data.has_baseline ? t('insights.why.learning') : main ? t(`insights.why.head.${main.verdict}`, { name: t(`insights.purpose.${main.id}`) }) : t('insights.why.head.normal')}
          </p>
          {data.anomalies.map((a: any) => a.kind === 'provider_rejected' ? <p key={a.id} className="m-0 text-[13px]">{t('insights.anomaly.explain.provider_rejected')}</p> : null)}
          <div className="overflow-x-auto">
            <table className="w-full min-w-[480px] text-[13px]">
              <thead><tr className="text-muted-foreground border-b text-xs">
                <th className="p-1.5 text-left font-medium">{t('insights.why.purpose')}</th>
                <th className="p-1.5 text-right font-medium">{t('insights.tokens')}</th>
                <th className="p-1.5 text-right font-medium">{t('insights.why.work')}</th>
                <th className="p-1.5 text-right font-medium">{t('insights.why.perUnit')}</th>
                <th className="p-1.5 text-right font-medium">{t('insights.card.cache')}</th>
                <th className="p-1.5 text-left font-medium">{t('insights.why.verdict')}</th>
              </tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className="border-b align-top last:border-0">
                    <td className="p-1.5"><PurposeName id={r.id} /></td>
                    <td className="p-1.5 text-right">{fmtTokens(r.tokens)}<div className="text-muted-foreground text-xs">{r.usual !== null ? t('insights.why.vs', { value: fmtTokens(r.usual) }) : ''}</div></td>
                    <td className="p-1.5 text-right">{Math.round(r.work)}<div className="text-muted-foreground text-xs">{r.usual_work !== null ? t('insights.why.vs', { value: Math.round(r.usual_work) }) : ''}</div></td>
                    <td className="p-1.5 text-right">{fmtTokens(r.per_unit)}<div className="text-muted-foreground text-xs">{r.usual_per_unit !== null ? t('insights.why.vs', { value: fmtTokens(r.usual_per_unit) }) : ''}</div></td>
                    <td className="p-1.5 text-right">{fmtPct(r.cache_rate)}<div className="text-muted-foreground text-xs">{r.usual_cache_rate !== null ? t('insights.why.vs', { value: fmtPct(r.usual_cache_rate) }) : ''}</div></td>
                    <td className="p-1.5 font-medium">{t(`insights.why.verdicts.${r.verdict}`)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.has_baseline ? (
            <div className="pb-9">
              <h3 className="m-0 mb-2 text-sm font-semibold">{t('insights.why.steps')}</h3>
              <WhySteps rows={rows} />
            </div>
          ) : null}
          <div>
            <h3 className="m-0 mb-1 text-sm font-semibold">{t('insights.why.hours')}</h3>
            <HoursChart hours={data.hours} purposes={[...PURPOSES]} />
          </div>
          <div>
            <h3 className="m-0 mb-1 text-sm font-semibold">{t('insights.why.events')}</h3>
            {data.events.length ? data.events.map((e: any, i: number) => (
              <div key={i} className="grid items-baseline gap-2 py-1 text-[13px]" style={{ gridTemplateColumns: '48px minmax(0, 1fr)' }}>
                <span className="text-muted-foreground text-xs">{e.at}</span>
                <span>{t(`insights.event.${e.kind}`, { defaultValue: e.kind })}{e.purpose ? ` · ${t(`insights.purpose.${e.purpose}`)}` : ''}{e.workspace ? ` · ${e.workspace}` : ''}{e.detail?.new ? ` · ${e.detail.old} → ${e.detail.new}` : ''}</span>
              </div>
            )) : <p className="text-muted-foreground m-0 text-[13px]">{t('insights.why.noEvents')}</p>}
          </div>
          {data.workspaces.length ? (
            <div>
              <h3 className="m-0 mb-1 text-sm font-semibold">{t('insights.why.workspaces')}</h3>
              {data.workspaces.map((w: any) => (
                <RowButton key={w.id} onClick={() => { goExplore('purpose', [{ dim: 'day', key: day, label: fmtDay(day) }, { dim: 'workspace', key: w.id, label: w.id }]); set({ day: null }) }}>
                  <span className="flex justify-between gap-3 text-[13px]"><span className="break-all">{w.id}</span><strong>{w.delta >= 0 ? '+' : '−'}{fmtTokens(Math.abs(w.delta))}</strong></span>
                </RowButton>
              ))}
            </div>
          ) : null}
          <div>
            <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => { goExplore('purpose', [{ dim: 'day', key: day, label: fmtDay(day) }]); set({ day: null }) }}>{t('insights.why.explore')}</button>
          </div>
        </>
      )}
    </motion.aside>
  )
}

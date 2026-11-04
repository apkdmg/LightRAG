import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { insightsApi } from './api'
import { useData, useInsights } from './context'
import { Card, Empty, RowButton, Seg, Title } from '@/components/admin'
import { PurposeName, SeverityLabel, fmtDay, fmtMoney, fmtTokens } from './ui'

const RANK: Record<string, number> = { critical: 3, serious: 2, warning: 1 }

export function anomalyTitle(t: any, a: any) {
  return t(`insights.anomaly.kind.${a.kind}`, { model: a.detail?.model, waiting: a.detail?.waiting, defaultValue: a.kind })
}

const when = (t: any, a: any) => {
  const days: string[] = a.run_days ?? [a.day]
  return days.length > 1 ? t('insights.anomalies.run', { first: fmtDay(days[0]), last: fmtDay(days[days.length - 1]), count: days.length }) : fmtDay(a.day, true)
}

export function isByo(a: any) {
  return a.detail?.byo === true
}

function Calendar({ cells, onPick, name }: { cells: any[]; onPick: (day: string) => void; name: string }) {
  const { t } = useTranslation()
  const max = Math.max(...cells.map((c) => c.tokens), 1)
  const offset = cells.length ? new Date(cells[0].day + 'T00:00:00').getDay() : 0
  return (
    <Card>
      <Title sub={t('insights.anomalies.calendarSub')}>{t('insights.anomalies.calendar', { name })}</Title>
      <div className="inline-grid grid-flow-col gap-[3px]" style={{ gridTemplateRows: 'repeat(7, 15px)' }}>
        {Array.from({ length: offset }).map((_, i) => <span key={i} />)}
        {cells.map((c) => {
          const step = c.tokens ? Math.min(5, 1 + Math.floor(Math.sqrt(c.tokens / max) * 5)) : 1
          return (
            <button
              key={c.day}
              type="button"
              title={`${fmtDay(c.day, true)}, ${fmtTokens(c.tokens)}${c.severity ? `, ${t(`insights.severity.${c.severity}`)}` : ''}`}
              onClick={() => onPick(c.day)}
              className="size-[15px] rounded-[3px]"
              style={{ background: `var(--ti-h${step})`, outline: c.severity ? '2px solid #d03b3b' : undefined, outlineOffset: -1 }}
            />
          )
        })}
      </div>
    </Card>
  )
}

export default function AnomaliesTab() {
  const { t } = useTranslation()
  const { scope, set, refreshHeader } = useInsights()
  const reduced = useReducedMotion()
  const [filter, setFilter] = useState<'open' | 'handled'>('open')
  const [selected, setSelected] = useState<string | null>(null)
  const { data, error, reload } = useData(
    () => insightsApi.anomalies({ workspace: scope.workspace, purpose: scope.purpose }),
    [scope.workspace, scope.purpose]
  )
  if (error && !data) return <Empty>{error}</Empty>
  if (!data) return <Empty>{t('insights.loading')}</Empty>
  if (!data.ready) {
    return (
      <Card className="flex max-w-[640px] flex-col gap-2.5">
        <Title sub={t('insights.anomalies.learningSub')}>{t('insights.anomalies.learning')}</Title>
        <div className="text-xs font-semibold">{t('insights.anomalies.daysLeft', { count: data.days_left })}</div>
      </Card>
    )
  }
  const items: any[] = data.items
  const open = items.filter((a) => a.state === 'open')
  const handled = items.filter((a) => a.state !== 'open')
  const shown = (filter === 'open' ? open : handled).sort((a, b) => RANK[b.severity] - RANK[a.severity] || (a.day < b.day ? 1 : -1))
  const current = shown.find((a) => a.id === selected) ?? shown[0]
  const handle = async (a: any, state: string) => {
    await insightsApi.setAnomaly(a.id, state)
    setSelected(null)
    await reload()
    refreshHeader()
  }
  const lens = scope.purpose ? t(`insights.purpose.${scope.purpose}`) : t('insights.anomalies.allPurposes')
  return (
    <div className="flex flex-col gap-4">
      <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(380px, 100%), 1fr))' }}>
        <Card>
          <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
            <Title>{t('insights.anomalies.inbox')}</Title>
            <Seg value={filter} onChange={(v) => { setFilter(v); setSelected(null) }} options={[{ value: 'open', label: t('insights.anomalies.open', { count: open.length }) }, { value: 'handled', label: t('insights.anomalies.handled', { count: handled.length }) }]} />
          </div>
          <AnimatePresence initial={false}>
            {shown.map((a) => (
              <motion.div key={a.id} layout={!reduced} exit={reduced ? undefined : { opacity: 0, height: 0 }} transition={{ duration: 0.2 }}>
                <RowButton active={current?.id === a.id} onClick={() => setSelected(a.id)}>
                  <span className="mb-0.5 flex flex-wrap items-center justify-between gap-2">
                    <strong className="text-[13px]">{anomalyTitle(t, a)}</strong>
                    <span className={a.severity === 'critical' && a.state === 'open' ? 'ti-pulse rounded-full' : ''}><SeverityLabel severity={a.severity} /></span>
                  </span>
                  <span className="text-muted-foreground block text-xs">
                    {when(t, a)}{a.purposes[0]?.ratio ? ` · ${t(`insights.purpose.${a.purposes[0].id}`)} ${a.purposes[0].ratio >= 1 ? t('insights.anomalies.times', { ratio: a.purposes[0].ratio.toFixed(1) }) : t('insights.anomalies.down', { pct: Math.round((1 - a.purposes[0].ratio) * 100) })}` : ''}{a.workspace ? ` · ${a.workspace}` : scope.workspace ? ` · ${t('insights.workspacePage.allWorkspaces')}` : ''}{isByo(a) ? ' · BYO' : ''}
                  </span>
                </RowButton>
              </motion.div>
            ))}
          </AnimatePresence>
          {!shown.length ? <Empty>{t(filter === 'open' ? 'insights.anomalies.noneOpen' : 'insights.anomalies.noneHandled')}</Empty> : null}
        </Card>
        {current ? (
          <Card className="flex flex-col gap-2.5">
            <div className="flex items-start justify-between gap-2.5">
              <Title sub={`${when(t, current)}${current.workspace ? ` · ${current.workspace}` : ''}`}>{anomalyTitle(t, current)}</Title>
              <SeverityLabel severity={current.severity} />
            </div>
            <p className="m-0 text-[13px]">{t(`insights.anomaly.explain.${current.kind}`, { defaultValue: '', model: current.detail?.model, waiting: current.detail?.waiting })}</p>
            {current.purposes.map((p: any) => (
              <div key={p.id} className="flex items-center justify-between gap-3 text-[13px]">
                <PurposeName id={p.id} />
                <span>{p.ratio ? `${fmtTokens(p.tokens)} · ${t('insights.anomalies.usually', { value: fmtTokens(p.usual) })}` : '–'}</span>
              </div>
            ))}
            {current.extra_cost ? <div className="text-[13px] font-semibold">{t('insights.anomalies.extra', { value: fmtMoney(current.extra_cost) })}</div> : null}
            {current.events?.length ? (
              <div className="text-muted-foreground text-xs">
                {current.events.map((e: any, i: number) => <div key={i}>{fmtDay(e.day)} {e.at} · {t(`insights.event.${e.kind}`, { defaultValue: e.kind })}</div>)}
              </div>
            ) : null}
            <div className="flex flex-wrap gap-2">
              <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => set({ day: (current.run_days ?? [current.day])[0] })}>{t('insights.anomalies.openDay')}</button>
              {current.state === 'open' ? (
                <>
                  <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => handle(current, 'expected')}>{t('insights.anomalies.expected')}</button>
                  <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => handle(current, 'acknowledged')}>{t('insights.anomalies.ack')}</button>
                </>
              ) : (
                <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => handle(current, 'open')}>{t('insights.anomalies.reopen')}</button>
              )}
            </div>
          </Card>
        ) : null}
      </div>
      <Calendar cells={data.calendar} onPick={(day) => set({ day })} name={`${lens}${scope.workspace ? `, ${scope.workspace}` : ''}`} />
    </div>
  )
}

import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { insightsApi } from './api'
import { UsageChart, ChartPoint } from './charts'
import { useData, useInsights } from './context'
import { INGEST, PURPOSES, PURPOSE_COLOR, Purpose, QUESTIONS, isFarFromOthers } from './meta'
import PurposePage from './PurposePage'
import WorkspacePage from './WorkspacePage'
import { Card, Empty, Kv, RowButton, Seg, Spark, Tile, Title } from '@/components/admin'
import { Counter, Mix, PurposeName, StatusLabel, Swatch, fmtChange, fmtCount, fmtDay, fmtMoney, fmtPct, fmtTokens, markIntroPlayed } from './ui'

export function buildPoints(chart: any, unit: 'tokens' | 'cost', bucket: 'day' | 'week'): ChartPoint[] {
  if (!chart) return []
  const raw: ChartPoint[] = chart.points.map((p: any) => ({
    key: p.key,
    day: p.day,
    hour: p.hour,
    partial: p.partial,
    future: p.future,
    gap: p.gap,
    label: chart.grain === 'hour' ? `${String(p.hour).padStart(2, '0')}:00` : fmtDay(p.day),
    values: unit === 'cost' ? p.cost : p.tokens,
    usual: p.usual,
    anomaly: p.anomaly,
    pins: p.pins
  }))
  if (bucket !== 'week' || chart.grain === 'hour') {
    return raw.map((p) => ({ ...p, label: chart.grain === 'hour' ? p.label : fmtDay(p.day, raw.length <= 14) }))
  }
  const weeks: ChartPoint[] = []
  for (let i = 0; i < raw.length; i += 7) {
    const group = raw.slice(i, i + 7)
    const values: Record<string, number> = {}
    for (const p of group) for (const q of PURPOSES) values[q] = (values[q] || 0) + (p.values[q] || 0)
    const last = group[group.length - 1]
    weeks.push({
      key: last.key,
      day: last.day,
      label: fmtDay(group[0].day),
      partial: last.partial,
      values,
      usual: group.every((p) => p.usual !== null && p.usual !== undefined) ? group.reduce((a, p) => a + (p.usual || 0), 0) : null,
      anomaly: group.find((p) => p.anomaly)?.anomaly,
      pins: group.flatMap((p) => p.pins || [])
    })
  }
  return weeks
}

export function ChartCard({ data, purposes }: { data: any; purposes: Purpose[] }) {
  const { t } = useTranslation()
  const { scope, set } = useInsights()
  const [layout, setLayout] = useState<'combined' | 'separate'>('combined')
  const [bucket, setBucket] = useState<'day' | 'week'>('day')
  const days = data.scope.range
  const showBuckets = days > 7
  const points = useMemo(
    () => buildPoints(data.chart, scope.unit, showBuckets ? bucket : 'day'),
    [data.chart, scope.unit, bucket, showBuckets]
  )
  const per = days === 1 ? 'hour' : bucket === 'week' && showBuckets ? 'week' : 'day'
  return (
    <Card>
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2.5">
        <Title sub={t('insights.chart.hint')}>
          {t(`insights.chart.title.${scope.unit}.${per}`, { scope: scope.workspace ? `${scope.workspace}: ` : '' })}
        </Title>
        <div className="flex flex-wrap gap-2">
          {purposes.length > 1 ? (
            <Seg value={layout} onChange={setLayout} options={[{ value: 'combined', label: t('insights.chart.combined') }, { value: 'separate', label: t('insights.chart.separate') }]} />
          ) : null}
          {showBuckets ? (
            <Seg value={bucket} onChange={setBucket} options={[{ value: 'day', label: t('insights.chart.perDay') }, { value: 'week', label: t('insights.chart.perWeek') }]} />
          ) : null}
        </div>
      </div>
      <UsageChart
        points={points}
        purposes={purposes}
        unit={scope.unit}
        layout={purposes.length > 1 ? layout : 'combined'}
        onPick={data.chart.grain === 'day' ? (day) => set({ day }) : () => set({ day: data.scope.end })}
      />
      {purposes.length > 1 && layout === 'combined' ? (
        <div className="mt-2 flex flex-wrap gap-1">
          {purposes.map((p) => {
            const row = data.purposes.find((x: any) => x.id === p)
            return (
              <button key={p} type="button" onClick={() => set({ purpose: p })} className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg px-2.5 py-1 text-[13px]">
                <Swatch purpose={p} />
                {t(`insights.purpose.${p}`)}
                <span className="text-muted-foreground">{fmtPct(row?.share_tokens)}</span>
              </button>
            )
          })}
        </div>
      ) : null}
    </Card>
  )
}

export function PurposeCards({ data }: { data: any }) {
  const { t } = useTranslation()
  const { set } = useInsights()
  const learn = !data.learning.anomalies_ready
  const groups: [string, Purpose[]][] = [['ingest', INGEST], ['questions', QUESTIONS]]
  return (
    <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(420px, 100%), 1fr))' }}>
      {groups.map(([name, ids]) => (
        <div key={name} className="min-w-0">
          <div className="mb-2 flex flex-wrap justify-between gap-2">
            <strong className="text-[13px]">{t(`insights.group.${name}.title`)}</strong>
            <span className="text-muted-foreground text-xs">{t(`insights.group.${name}.note`)}</span>
          </div>
          <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(195px, 100%), 1fr))' }}>
            {ids.map((id) => {
              const p = data.purposes.find((x: any) => x.id === id)
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => set({ purpose: id })}
                  className="bg-card hover:border-muted-foreground/40 flex min-w-0 flex-col gap-1.5 rounded-[10px] border p-3.5 text-left transition-all hover:-translate-y-px hover:shadow-md"
                >
                  <span className="flex items-center gap-1.5"><Swatch purpose={id} /><strong className="text-[15px]">{t(`insights.purpose.${id}`)}</strong></span>
                  <StatusLabel status={p.status} />
                  <span className="text-[28px] leading-tight font-semibold tracking-tight"><Counter value={p.tokens} format={fmtTokens} /></span>
                  <span className="text-muted-foreground text-xs">{t('insights.card.shares', { tokens: fmtPct(p.share_tokens), spend: fmtPct(p.share_cost) })}</span>
                  <Spark values={p.series} color={PURPOSE_COLOR[id]} />
                  <Kv k={t(`insights.card.per.${id}`)}>{fmtTokens(p.per_unit)}</Kv>
                  <Kv k={t('insights.card.usually')}>{learn || p.per_unit_usual === null ? t('insights.notKnownYet') : fmtTokens(p.per_unit_usual)}</Kv>
                  {data.scope.workspace && p.others_per_unit ? (
                    <Kv k={t('insights.card.others')}>
                      {fmtTokens(p.others_per_unit)}
                      {isFarFromOthers(p.others_ratio) ? ` · ${t('insights.card.thisOne', { ratio: p.others_ratio.toFixed(1) })}` : ''}
                    </Kv>
                  ) : null}
                  <Kv k={t('insights.card.cache')}>{fmtPct(p.cache_rate)}</Kv>
                  <span className="text-muted-foreground truncate font-mono text-xs">
                    {p.model || '–'}{p.billing === 'byo' ? ' (BYO)' : ''}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

function ShareBars({ data }: { data: any }) {
  const { t } = useTranslation()
  const rows = data.purposes
  const topTokens = [...rows].sort((a: any, b: any) => b.share_tokens - a.share_tokens)[0]
  const topCost = [...rows].sort((a: any, b: any) => b.share_cost - a.share_cost)[0]
  const sentence = topTokens.id === topCost.id
    ? t('insights.shares.same', { name: t(`insights.purpose.${topTokens.id}`) })
    : t('insights.shares.different', { tokens: t(`insights.purpose.${topTokens.id}`), tokensPct: fmtPct(topTokens.share_tokens), cost: t(`insights.purpose.${topCost.id}`), costPct: fmtPct(topCost.share_cost) })
  const bar = (label: string, key: string) => (
    <div className="mb-3.5">
      <div className="mb-1 text-xs font-semibold">{label}</div>
      <Mix tall values={Object.fromEntries(rows.map((r: any) => [r.id, r[key]]))} />
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
        {rows.map((r: any) => (
          <span key={r.id} className="inline-flex items-center gap-1.5"><Swatch purpose={r.id} />{t(`insights.purpose.${r.id}`)} <strong>{fmtPct(r[key])}</strong></span>
        ))}
      </div>
    </div>
  )
  return (
    <Card>
      <Title sub={sentence}>{t('insights.shares.title')}</Title>
      <div className="mt-3">{bar(t('insights.shares.tokens'), 'share_tokens')}{bar(t('insights.shares.spend'), 'share_cost')}</div>
    </Card>
  )
}

function ModelTable({ models }: { models: any[] }) {
  const { t } = useTranslation()
  const { goExplore } = useInsights()
  const used = models.filter((m) => m.tokens > 0)
  const top: Record<string, string> = {}
  for (const p of PURPOSES) {
    top[p] = [...models].sort((a, b) => (b.by_purpose[p] || 0) - (a.by_purpose[p] || 0))[0]?.id
  }
  return (
    <Card>
      <Title sub={t('insights.models.sub')}>{t('insights.models.title')}</Title>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-[13px]">
          <thead>
            <tr className="text-muted-foreground border-b text-xs">
              <th className="p-2 text-left font-medium">{t('insights.models.model')}</th>
              {PURPOSES.map((p) => <th key={p} className="p-2 text-right font-medium"><PurposeName id={p} /></th>)}
              <th className="p-2 text-right font-medium">{t('insights.tokens')}</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const badge = m === used[0] ? 'most' : m === used[used.length - 1] && used.length > 1 ? 'least' : !m.tokens ? 'idle' : null
              return (
                <tr key={m.id} className="border-b last:border-0">
                  <td className="p-2">
                    <span className="block font-mono text-xs">{m.id || t('insights.models.unknown')}{m.byo ? ' (BYO)' : ''}</span>
                    {badge ? <span className="text-muted-foreground mt-0.5 inline-block rounded-full border px-1.5 text-[11px]">{t(`insights.models.${badge}`)}</span> : null}
                  </td>
                  {PURPOSES.map((p) => (
                    <td key={p} className="p-1 text-right">
                      {m.by_purpose[p] ? (
                        <button
                          type="button"
                          className={`hover:bg-muted min-h-7 rounded px-2 ${top[p] === m.id ? 'font-bold' : ''}`}
                          onClick={() => goExplore('operation', [{ dim: 'model', key: m.id, label: m.id }], { purpose: p })}
                        >
                          {fmtPct(m.by_purpose[p])}
                        </button>
                      ) : <span className="text-muted-foreground">–</span>}
                    </td>
                  ))}
                  <td className="p-2 text-right font-semibold">{fmtTokens(m.tokens)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {!models.length ? <Empty>{t('insights.empty')}</Empty> : null}
      </div>
    </Card>
  )
}

function Ranking({ rows }: { rows: any[] }) {
  const { t } = useTranslation()
  const { set } = useInsights()
  return (
    <Card>
      <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
        <Title sub={t('insights.ranking.sub')}>{t('insights.ranking.title')}</Title>
        <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => set({ tab: 'workspaces' })}>{t('insights.ranking.all')}</button>
      </div>
      <div className="overflow-x-auto"><div className="min-w-[640px]">
        {rows.map((r, i) => {
          const asks = ((r.mix.keyword || 0) + (r.mix.query || 0)) / (Object.values<number>(r.mix).reduce((a, b) => a + b, 0) || 1)
          const kind = asks > 0.5 ? 'questions' : asks < 0.2 ? 'ingestion' : 'mixed'
          return (
            <RowButton key={r.id} onClick={r.deleted ? undefined : () => set({ workspace: r.id })}>
              <span className="grid items-center gap-2.5" style={{ gridTemplateColumns: '20px minmax(130px, 1fr) minmax(150px, 2fr) minmax(120px, 230px) 64px' }}>
                <span className="text-muted-foreground text-xs">{i + 1}</span>
                <span className="text-[13px] break-all">{r.deleted ? t('insights.workspaces.deleted', { code: r.id.replace('deleted-', '').slice(0, 4) }) : r.id}</span>
                <Mix values={r.mix} />
                <span className="text-muted-foreground text-right text-xs">
                  {t(`insights.ranking.${kind}`)}
                  {r.change !== null ? ` · ${fmtChange(r.change)}` : ''}
                  {r.byo_for.length ? ` · ${t('insights.ranking.byoFor', { names: r.byo_for.map((p: string) => t(`insights.purpose.${p}`)).join(', ') })}` : ''}
                </span>
                <strong className="text-right text-[13px]">{fmtTokens(r.tokens)}</strong>
              </span>
            </RowButton>
          )
        })}
        {!rows.length ? <Empty>{t('insights.empty')}</Empty> : null}
      </div></div>
    </Card>
  )
}

function Efficiency({ efficiency }: { efficiency: any }) {
  const { t } = useTranslation()
  const { headroom, wasted } = efficiency
  const used = headroom.limit_tpm ? headroom.peak_tpm / headroom.limit_tpm : null
  return (
    <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(360px, 100%), 1fr))' }}>
      <Card>
        <Title sub={t('insights.wasted.sub')}>{t('insights.wasted.title')}</Title>
        {PURPOSES.map((p) => {
          const w = wasted[p]
          return (
            <div key={p} className="flex items-center justify-between gap-3 border-b py-1.5 text-[13px] last:border-0">
              <PurposeName id={p} />
              <span>{w ? `${fmtTokens(w.tokens)} · ${fmtPct(w.share)}` : '–'}</span>
            </div>
          )
        })}
      </Card>
      <Card>
        <Title sub={t('insights.headroom.sub')}>{t('insights.headroom.title')}</Title>
        <Kv k={t('insights.headroom.peak')}>{t('insights.headroom.tpm', { value: fmtTokens(headroom.peak_tpm) })}</Kv>
        <Kv k={t('insights.headroom.limit')}>{headroom.limit_tpm ? t('insights.headroom.tpm', { value: fmtTokens(headroom.limit_tpm) }) : t('insights.headroom.notSet')}</Kv>
        {used !== null ? <Kv k={t('insights.headroom.used')}>{fmtPct(used)}</Kv> : null}
        <div className="mt-2 flex flex-wrap gap-x-3 text-xs">
          {Object.entries<number>(headroom.by_purpose).map(([p, v]) => (
            <span key={p} className="inline-flex items-center gap-1.5"><Swatch purpose={p as Purpose} />{t(`insights.purpose.${p}`)} <strong>{fmtTokens(v)}</strong></span>
          ))}
        </div>
      </Card>
    </div>
  )
}

export function Strip({ data }: { data: any }) {
  const { t } = useTranslation()
  const { totals, learning, scope, budget } = data
  const compare = learning.days_collected < 2 ? t('insights.notEnoughToCompare') : null
  const period = scope.range === 1 ? t('insights.vsUsual') : t('insights.vsPrevious', { days: scope.range })
  const costNotes = [compare ?? (totals.cost_change !== null ? `${fmtChange(totals.cost_change)} ${period}` : ''), totals.byo_cost > 0 ? t('insights.onByoKeys', { value: fmtMoney(totals.byo_cost) }) : '', totals.unpriced_tokens > 0 ? t('insights.noPrice', { value: fmtTokens(totals.unpriced_tokens) }) : ''].filter(Boolean).join(' · ')
  return (
    <Card className="flex flex-wrap items-end gap-x-7 gap-y-4">
      <Tile hero label={t('insights.strip.tokens')} value={<Counter value={totals.tokens} format={fmtTokens} />} note={compare ?? (totals.tokens_change !== null ? `${fmtChange(totals.tokens_change)} ${period}` : '')} />
      <Tile label={t('insights.strip.spend')} value={<Counter value={totals.cost} format={fmtMoney} />} note={costNotes} />
      <Tile label={t('insights.strip.saved')} value={<Counter value={totals.saved_tokens} format={fmtTokens} />} note={t('insights.strip.savedNote')} />
      <Tile
        label={t('insights.strip.budget')}
        value={!learning.forecast_ready ? t('insights.notKnownYet') : !budget?.set ? t('insights.strip.noBudget') : budget.runs_out ? fmtDay(budget.runs_out) : t('insights.strip.later')}
        note={!learning.forecast_ready ? t('insights.needs28') : budget?.days_left !== undefined && budget?.days_left !== null ? t('insights.strip.daysLeft', { count: budget.days_left }) : ''}
      />
    </Card>
  )
}

export default function OverviewTab() {
  const { t } = useTranslation()
  const { scope, params } = useInsights()
  const { data, loading, error } = useData(
    () => insightsApi.overview({ ...params, purpose: scope.purpose, workspace: scope.workspace }),
    [JSON.stringify(params), scope.purpose, scope.workspace],
    60
  )
  useEffect(() => {
    if (data) {
      const timer = setTimeout(markIntroPlayed, 800)
      return () => clearTimeout(timer)
    }
  }, [data])
  if (error && !data) return <Empty>{error}</Empty>
  if (!data) return <Empty>{t('insights.loading')}</Empty>
  const dim = loading ? 'opacity-60 transition-opacity' : 'transition-opacity'
  if (data.scope.purpose !== (scope.purpose || null) || data.scope.workspace !== (scope.workspace || null)) {
    return <div className={dim}><Empty>{t('insights.loading')}</Empty></div>
  }
  if (scope.purpose) return <div className={dim}><PurposePage data={data} /></div>
  if (scope.workspace) return <div className={dim}><WorkspacePage data={data} /></div>
  return (
    <div className={`flex flex-col gap-4 ${dim}`}>
      <Strip data={data} />
      <PurposeCards data={data} />
      <ChartCard data={data} purposes={[...PURPOSES]} />
      <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(380px, 100%), 1fr))' }}>
        <ShareBars data={data} />
        <ModelTable models={data.models} />
      </div>
      <Ranking rows={data.workspaces} />
      <Efficiency efficiency={data.efficiency} />
      {data.quality.not_measured_calls ? <p className="text-muted-foreground m-0 text-xs">{t('insights.notMeasured', { count: fmtCount(data.quality.not_measured_calls) })}</p> : null}
    </div>
  )
}

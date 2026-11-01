import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import { insightsApi } from './api'
import { useData, useInsights } from './context'
import { PURPOSE_COLOR, PURPOSE_SETTINGS, Purpose } from './meta'
import { ChartCard } from './OverviewTab'
import { Card, Counter, Empty, Kv, Meter, RowButton, StatusLabel, Swatch, TextButton, Tile, Title, fmtChange, fmtCount, fmtDay, fmtMoney, fmtPct, fmtTokens } from './ui'

function ChangeCheck({ change, unit }: { change: any; unit: string }) {
  const { t } = useTranslation()
  if (!change) return <p className="m-0 text-[13px]">{t('insights.change.none')}</p>
  if (!change.ready) {
    return <p className="m-0 text-[13px]">{t('insights.change.pending', { day: fmtDay(change.day), to: change.to, ready: fmtDay(change.ready_on) })}</p>
  }
  const row = (label: string, a: number, b: number, fmt: (n: number) => string) => (
    <tr className="border-b last:border-0">
      <td className="p-1.5">{label}</td>
      <td className="p-1.5 text-right">{fmt(a)}</td>
      <td className="p-1.5 text-right">{fmt(b)}</td>
      <td className="p-1.5 text-right font-semibold">{fmtChange((b - a) / (a || 1))}</td>
    </tr>
  )
  const costChange = (change.after.cost - change.before.cost) / (change.before.cost || 1)
  return (
    <>
      <p className="text-muted-foreground m-0 mb-1 text-[13px]">{t('insights.change.what', { day: fmtDay(change.day), from: change.from, to: change.to })}</p>
      <table className="w-full text-[13px]">
        <thead><tr className="text-muted-foreground border-b text-xs"><th className="p-1.5 text-left font-medium">{t('insights.change.per', { unit })}</th><th className="p-1.5 text-right font-medium">{t('insights.change.before')}</th><th className="p-1.5 text-right font-medium">{t('insights.change.after')}</th><th className="p-1.5 text-right font-medium">{t('insights.change.change')}</th></tr></thead>
        <tbody>
          {row(t('insights.tokens'), change.before.tokens, change.after.tokens, fmtTokens)}
          {row(t('insights.cost'), change.before.cost, change.after.cost, fmtMoney)}
        </tbody>
      </table>
      <p className="m-0 mt-2 text-[13px] font-medium">{t(costChange > 0.05 ? 'insights.change.costsMore' : costChange < -0.05 ? 'insights.change.costsLess' : 'insights.change.same', { pct: Math.abs(Math.round(costChange * 100)) })}</p>
    </>
  )
}

export default function PurposePage({ data }: { data: any }) {
  const { t } = useTranslation()
  const { scope, set, params, goExplore } = useInsights()
  const id = scope.purpose as Purpose
  const { data: page, error } = useData(
    () => insightsApi.purpose(id, { ...params, workspace: scope.workspace }),
    [id, JSON.stringify(params), scope.workspace]
  )
  const row = data.purposes.find((p: any) => p.id === id)
  const color = PURPOSE_COLOR[id]
  const unit = t(`insights.unit.${id}`)
  if (error) return <Empty>{error}</Empty>
  const totals = data.totals
  const usualPer = row.per_unit_usual
  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-wrap items-start justify-between gap-x-7 gap-y-2.5">
        <div className="min-w-0 flex-[1_1_340px]">
          {scope.workspace ? <TextButton className="mb-1 text-[13px]" onClick={() => set({ purpose: '' })}>← {t('insights.purposePage.back', { workspace: scope.workspace })}</TextButton> : null}
          <div className="flex items-center gap-2"><Swatch purpose={id} size={14} /><h2 className="m-0 text-[22px] font-semibold tracking-tight">{scope.workspace ? t('insights.purposePage.inWorkspace', { name: t(`insights.purpose.${id}`), workspace: scope.workspace }) : t(`insights.purpose.${id}`)}</h2></div>
          <p className="text-muted-foreground mt-1.5 text-[15px]">{t(`insights.purposePage.desc.${id}`)}</p>
        </div>
        <div className="flex min-w-0 flex-[0_1_380px] flex-col gap-1">
          <Kv k={t('insights.purposePage.runs')}>{t(`insights.purposePage.when.${id}`)}</Kv>
          <Kv k={t('insights.purposePage.model')}><span className="font-mono">{page?.model || row.model || '–'}{(page?.billing || row.billing) === 'byo' ? ' (BYO)' : ''}</span></Kv>
          <Kv k={t('insights.purposePage.slot')}>{t(`insights.purposePage.slotName.${id}`)}</Kv>
          <Kv k={t('insights.purposePage.onByo')}>{page ? `${fmtPct(page.byo_share)}${page.byo_workspaces.length ? ` · ${page.byo_workspaces.join(', ')}${page.byo_workspace_count > page.byo_workspaces.length ? ` ${t('insights.purposePage.andMore', { count: page.byo_workspace_count - page.byo_workspaces.length })}` : ''}` : ''}` : '–'}</Kv>
          <Kv k={t('insights.purposePage.today')}><StatusLabel status={row.status} /></Kv>
        </div>
      </Card>
      <Card className="flex flex-wrap items-end gap-x-7 gap-y-4">
        <Tile hero label={t('insights.purposePage.tokens', { name: t(`insights.purpose.${id}`) })} value={<Counter value={row.tokens} format={fmtTokens} />} note={totals.tokens_change !== null ? `${fmtChange(totals.tokens_change)} ${data.scope.range === 1 ? t('insights.vsUsual') : t('insights.vsPrevious', { days: data.scope.range })}` : ''} />
        <Tile label={t('insights.strip.spend')} value={<Counter value={row.cost} format={fmtMoney} />} note={row.byo_cost > 0 ? t('insights.onByoKeys', { value: fmtMoney(row.byo_cost) }) : ''} />
        <Tile label={t(`insights.purposePage.done.${id}`)} value={fmtCount(row.work)} note={totals.usual_work ? t('insights.usually', { value: fmtCount(totals.usual_work) }) : ''} />
        <Tile label={t(`insights.card.per.${id}`)} value={fmtTokens(row.per_unit)} note={usualPer ? t('insights.usually', { value: fmtTokens(usualPer) }) : ''} />
        <Tile label={t('insights.card.cache')} value={fmtPct(row.cache_rate)} note={t('insights.purposePage.cacheNote')} />
      </Card>
      <ChartCard data={data} purposes={[id]} />
      {page ? (
        <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(360px, 100%), 1fr))' }}>
          <Card>
            <Title sub={t(`insights.purposePage.opsNote.${id}`)}>{t('insights.purposePage.spentOn')}</Title>
            {page.operations.map((o: any) => (
              <RowButton key={o.id} onClick={() => goExplore('workspace', [{ dim: 'operation', key: o.id, label: t(`insights.operation.${o.id}`, { defaultValue: o.id }) }])}>
                <span className="flex justify-between gap-2 text-[13px]"><span>{t(`insights.operation.${o.id}`, { defaultValue: o.id })}</span><strong>{fmtTokens(o.tokens)}</strong></span>
                <span className="mt-1 block"><Meter value={(o.tokens / (page.operations[0].tokens || 1)) * 100} color={color} /></span>
              </RowButton>
            ))}
            {!page.operations.length ? <Empty>{t('insights.empty')}</Empty> : null}
          </Card>
          <Card>
            <Title sub={t('insights.purposePage.workspacesSub', { unit })}>{t('insights.purposePage.workspaces', { name: t(`insights.purpose.${id}`) })}</Title>
            {page.workspaces.map((w: any) => (
              <RowButton key={w.id} active={scope.workspace === w.id} onClick={w.id.startsWith('deleted-') ? undefined : () => set({ workspace: scope.workspace === w.id ? '' : w.id })}>
                <span className="flex justify-between gap-2 text-[13px]"><span className="break-all">{w.id}</span><strong>{fmtTokens(w.tokens)}</strong></span>
                <span className="my-1 block"><Meter value={(w.tokens / (page.workspaces[0].tokens || 1)) * 100} color={color} /></span>
                <span className="text-muted-foreground text-xs">{t('insights.purposePage.perUnit', { value: fmtTokens(w.per_unit), unit })}{w.ratio >= 1.25 ? ` · ${t('insights.purposePage.above')}` : w.ratio !== null && w.ratio <= 0.85 ? ` · ${t('insights.purposePage.belowOthers')}` : ''}</span>
              </RowButton>
            ))}
            {page.workspace_count > page.workspaces.length ? <button type="button" className="hover:bg-muted mt-2 min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => set({ tab: 'workspaces' })}>{t('insights.purposePage.allWorkspaces', { count: page.workspace_count })}</button> : null}
          </Card>
          <Card>
            <Title sub={t('insights.purposePage.modelsSub')}>{t('insights.purposePage.models', { name: t(`insights.purpose.${id}`) })}</Title>
            {page.models.map((m: any, i: number) => (
              <RowButton key={`${m.id}-${m.byo}`} onClick={() => goExplore('operation', [{ dim: 'model', key: m.id, label: m.id }])}>
                <span className="flex justify-between gap-2 text-[13px]"><span className="font-mono text-xs break-all">{m.id}{m.byo ? ' (BYO)' : ''}{page.models.length > 1 && i === 0 ? <span className="text-muted-foreground ms-1.5 rounded-full border px-1.5 font-sans text-[11px]">{t('insights.purposePage.mostUsed')}</span> : null}</span><strong>{fmtPct(m.share)}</strong></span>
                <span className="my-1 block"><Meter value={(m.share / (page.models[0].share || 1)) * 100} color={color} /></span>
                <span className="text-muted-foreground text-xs">{fmtTokens(m.tokens)} {t('insights.tokensLower')}{m.latency_avg_ms ? ` · ${(m.latency_avg_ms / 1000).toFixed(1)} s` : ''} · {t('insights.purposePage.errors', { value: fmtPct(m.error_rate) })}</span>
              </RowButton>
            ))}
          </Card>
          <Card className="flex flex-col gap-2">
            <Title>{t('insights.purposePage.rightNow')}</Title>
            <Kv k={t('insights.purposePage.todaySoFar')}>{fmtTokens(page.health.tokens_today)}</Kv>
            <Kv k={t('insights.purposePage.workers')}>{page.health.queue ? t('insights.purposePage.workersValue', { running: page.health.queue.running, max: page.health.queue.max_async, queued: page.health.queue.queued }) : '–'}</Kv>
            <Kv k={t('insights.purposePage.p95')}>{page.health.p95_ms ? `${(page.health.p95_ms / 1000).toFixed(1)} s` : '–'}</Kv>
            <Kv k={t('insights.purposePage.errorsToday')}>{fmtPct(page.health.error_rate)}</Kv>
            <Kv k={t('insights.purposePage.cutOff')}>{fmtPct(page.health.truncated_rate)}</Kv>
            {id === 'vlm' && page.health.not_measured ? <p className="text-muted-foreground m-0 text-xs">{t('insights.purposePage.notMeasured', { count: page.health.not_measured })}</p> : null}
            <h3 className="m-0 mt-2.5 text-[15px] font-semibold">{t('insights.purposePage.split')}</h3>
            <SplitBar split={page.split} color={color} />
          </Card>
          <Card className="flex flex-col gap-2">
            <Title>{t('insights.purposePage.lastChange')}</Title>
            <ChangeCheck change={page.model_change} unit={unit} />
          </Card>
          <Card className="flex flex-col gap-2.5">
            <Title>{t('insights.purposePage.canChange')}</Title>
            {PURPOSE_SETTINGS[id].map((s) => (
              <div key={s}><div className="font-mono text-xs font-semibold">{s}</div><div className="text-muted-foreground text-xs">{t(`insights.setting.${s}`)}</div></div>
            ))}
            <div><button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => useSettingsStore.getState().setCurrentTab('providers')}>{t('insights.openProviders')}</button></div>
          </Card>
        </div>
      ) : <Empty>{t('insights.loading')}</Empty>}
    </div>
  )
}

function SplitBar({ split, color }: { split: any; color: string }) {
  const { t } = useTranslation()
  const parts = [
    { key: 'prompt', value: split.prompt, color },
    { key: 'answer', value: split.answer, color: 'var(--muted-foreground)' },
    { key: 'reasoning', value: split.reasoning, color: 'var(--ti-axis)' }
  ].filter((p) => p.value > 0)
  const total = parts.reduce((a, p) => a + p.value, 0) || 1
  return (
    <>
      <span className="flex h-4 gap-0.5">{parts.map((p) => <span key={p.key} className="rounded-[2px]" style={{ flexGrow: p.value / total, background: p.color }} />)}</span>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
        {parts.map((p) => <span key={p.key} className="inline-flex items-center gap-1.5"><Swatch color={p.color} />{t(`insights.purposePage.part.${p.key}`)} <strong>{fmtPct(p.value / total)}</strong></span>)}
      </div>
      {split.provider_cached ? <p className="text-muted-foreground m-0 text-xs">{t('insights.purposePage.providerCached', { value: fmtTokens(split.provider_cached) })}</p> : null}
    </>
  )
}

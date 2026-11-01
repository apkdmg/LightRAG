import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import { downloadFile, insightsApi } from './api'
import { useData, useInsights } from './context'
import { PURPOSES, Purpose } from './meta'
import { ChartCard, PurposeCards } from './OverviewTab'
import { Card, Counter, Empty, Kv, Meter, RowButton, StatusLabel, Swatch, TextButton, Tile, Title, fmtChange, fmtCount, fmtDay, fmtMoney, fmtPct, fmtTokens } from './ui'

const ordinal = (n: number) => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th'][n % 10])

export function ItemName({ display }: { display: any }) {
  const { t } = useTranslation()
  if (display.kind === 'client') return <>{display.label}</>
  if (display.kind === 'deleted') return <>{t('insights.item.deleted', { code: display.code })}</>
  if (display.kind === 'email') return <>{t('insights.item.email', { code: display.code })}</>
  if (display.kind === 'email_attachment') return <>{t('insights.item.attachment', { code: display.code, name: display.label })}</>
  if (display.kind === 'email_image') return <>{t('insights.item.image', { code: display.code, name: display.label })}</>
  return <>{display.label} <span className="text-muted-foreground">#{display.code}</span></>
}

export default function WorkspacePage({ data }: { data: any }) {
  const { t } = useTranslation()
  const { scope, set, params, goExplore } = useInsights()
  const { data: page, error } = useData(
    () => insightsApi.workspace(scope.workspace, params),
    [scope.workspace, JSON.stringify(params)]
  )
  const { totals, learning } = data
  if (error) return <Empty>{error}</Empty>
  const period = data.scope.range === 1 ? t('insights.vsUsual') : t('insights.vsPrevious', { days: data.scope.range })
  const email = page?.owner && page.owner.includes('@') ? page.owner : null
  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-wrap items-start justify-between gap-x-9 gap-y-3.5">
        <div className="flex min-w-0 flex-[1_1_300px] flex-col gap-1.5">
          <TextButton className="text-[13px]" onClick={() => set({ workspace: '', tab: 'workspaces' })}>← {t('insights.workspacePage.all')}</TextButton>
          <h2 className="m-0 mb-1 text-[22px] font-semibold tracking-tight break-all">{scope.workspace}</h2>
          <Kv k={t('insights.workspacePage.owner')}>{page?.owner || t('insights.workspacePage.noOwner')}</Kv>
          <Kv k={t('insights.workspacePage.lastActive')}>{page?.last_active ? new Date(page.last_active).toLocaleString() : '–'}</Kv>
          <Kv k={t('insights.workspacePage.today')}>{page ? <StatusLabel status={page.status} /> : '–'}</Kv>
        </div>
        <div className="min-w-0 flex-[1_1_440px]">
          <div className="mb-1.5 text-xs font-semibold">{t('insights.workspacePage.whoPays')}</div>
          {page ? PURPOSES.map((p) => {
            const key = page.keys[p]
            return (
              <div key={p} className="grid items-baseline gap-2 border-b py-1 text-xs last:border-0" style={{ gridTemplateColumns: '96px minmax(0, 1fr) minmax(0, 1fr)' }}>
                <span className="inline-flex items-center gap-1.5"><Swatch purpose={p as Purpose} />{t(`insights.purpose.${p}`)}</span>
                <strong className="font-semibold">{key.billing === 'byo' ? (p === 'vlm' ? t('insights.workspacePage.ownKeyVision') : t('insights.workspacePage.ownKey')) : t('insights.workspacePage.systemKey')}</strong>
                <span className="text-muted-foreground font-mono break-all">{key.model || '–'}</span>
              </div>
            )
          }) : null}
        </div>
      </Card>
      <Card className="flex flex-wrap items-end gap-x-7 gap-y-4">
        <Tile hero label={t('insights.strip.tokens')} value={<Counter value={totals.tokens} format={fmtTokens} />} note={totals.tokens_change !== null ? `${fmtChange(totals.tokens_change)} ${period}` : ''} />
        <Tile label={t('insights.strip.spend')} value={<Counter value={totals.cost} format={fmtMoney} />} note={[totals.cost_change !== null ? `${fmtChange(totals.cost_change)} ${period}` : '', totals.byo_cost > 0 ? t('insights.onByoKeys', { value: fmtMoney(totals.byo_cost) }) : ''].filter(Boolean).join(' · ')} />
        <Tile label={t('insights.workspacePage.share')} value={page ? fmtPct(page.share_cost) : '–'} note={page?.rank ? t('insights.workspacePage.rank', { rank: ordinal(page.rank), count: page.workspace_count }) : ''} />
        <Tile label={t('insights.workspacePage.documents')} value={page ? fmtCount(page.documents) : '–'} note={learning.anomalies_ready && page ? t('insights.usually', { value: fmtCount(page.usual_documents) }) : ''} />
        <Tile label={t('insights.workspacePage.questions')} value={page ? fmtCount(page.questions) : '–'} note={learning.anomalies_ready && page ? t('insights.usually', { value: fmtCount(page.usual_questions) }) : ''} />
      </Card>
      <PurposeCards data={data} />
      <ChartCard data={data} purposes={[...PURPOSES]} />
      {page ? (
        <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(360px, 100%), 1fr))' }}>
          <Card className="flex flex-col gap-2">
            <Title>{t('insights.purposePage.rightNow')}</Title>
            <Kv k={t('insights.workspacePage.pipeline')}>{page.live.processing ? t('insights.workspacePage.processing', { count: page.live.processing }) : page.live.waiting ? t('insights.workspacePage.queued') : t('insights.workspacePage.idle')}</Kv>
            <Kv k={t('insights.workspacePage.waiting')}>{fmtCount(page.live.waiting)}</Kv>
            <Kv k={t('insights.workspacePage.failed')}>{fmtCount(page.live.failed)}</Kv>
            <Kv k={t('insights.workspacePage.processed')}>{fmtCount(page.live.processed)}</Kv>
            <Kv k={t('insights.workspacePage.questionsToday')}>{fmtCount(page.live.questions_today)}</Kv>
          </Card>
          <Card>
            <Title sub={t('insights.workspacePage.sourcesSub')}>{t('insights.workspacePage.sources')}</Title>
            {page.sources.map((s: any) => (
              <RowButton key={s.id} onClick={() => goExplore('purpose', [{ dim: 'source', key: s.id, label: t(`insights.source.${s.id}`, { defaultValue: s.id }) }])}>
                <span className="flex justify-between gap-2 text-[13px]"><span>{t(`insights.source.${s.id}`, { defaultValue: s.id })}</span><strong>{fmtTokens(s.tokens)}</strong></span>
                <span className="mt-1 block"><Meter value={(s.tokens / (page.sources[0].tokens || 1)) * 100} /></span>
              </RowButton>
            ))}
            {!page.sources.length ? <Empty>{t('insights.empty')}</Empty> : null}
          </Card>
          <Card>
            <Title sub={t('insights.workspacePage.itemsSub')}>{t('insights.workspacePage.items')}</Title>
            {page.items.map((i: any) => (
              <div key={`${i.kind}-${i.doc_id ?? i.item}`} className="flex justify-between gap-3 border-b py-1.5 text-[13px] last:border-0">
                <span className="break-all"><ItemName display={i.display} /> <span className="text-muted-foreground text-xs">{t(`insights.item.kind.${i.kind}`)}</span></span>
                <strong>{fmtTokens(i.tokens)}</strong>
              </div>
            ))}
            {!page.items.length ? <Empty>{t('insights.empty')}</Empty> : null}
            <button type="button" className="hover:bg-muted mt-2.5 min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => goExplore('document', [])}>{t('insights.workspacePage.seeAll')}</button>
          </Card>
          <Card>
            <Title sub={t('insights.workspacePage.eventsSub')}>{t('insights.workspacePage.events')}</Title>
            {page.events.map((e: any, n: number) => (
              <RowButton key={n} onClick={() => set({ day: e.day })}>
                <span className="grid items-center gap-2" style={{ gridTemplateColumns: '52px minmax(0, 1fr) auto' }}>
                  <span className="text-muted-foreground text-xs">{fmtDay(e.day)}</span>
                  <span className="text-[13px]">{t(e.type === 'anomaly' ? `insights.anomaly.kind.${e.kind}` : `insights.event.${e.kind}`, { defaultValue: e.kind })}</span>
                  <span className="text-muted-foreground rounded-full border px-1.5 text-[11px]">{e.shared ? t('insights.workspacePage.allWorkspaces') : e.type === 'anomaly' ? t(`insights.anomaly.state.${e.state}`) : t('insights.workspacePage.change')}</span>
                </span>
              </RowButton>
            ))}
            {!page.events.length ? <Empty>{t('insights.workspacePage.noEvents')}</Empty> : null}
          </Card>
          <Card className="flex flex-col gap-2.5">
            <Title sub={t('insights.workspacePage.actionsSub')}>{t('insights.workspacePage.actions')}</Title>
            <div className="flex flex-wrap gap-2">
              {email ? <a className="hover:bg-muted inline-flex min-h-9 items-center rounded-lg border px-3 text-[13px]" href={`mailto:${email}`}>{t('insights.workspacePage.email')}</a> : null}
              <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => useSettingsStore.getState().setCurrentTab('providers')}>{t('insights.openProviders')}</button>
              <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => downloadFile('/breakdown.csv', `${scope.workspace}-usage.csv`, { ...params, group: 'purpose', workspace: scope.workspace })}>{t('insights.workspacePage.export')}</button>
            </div>
          </Card>
        </div>
      ) : <Empty>{t('insights.loading')}</Empty>}
    </div>
  )
}

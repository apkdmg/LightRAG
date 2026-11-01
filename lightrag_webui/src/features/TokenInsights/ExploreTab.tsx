import { useTranslation } from 'react-i18next'
import { downloadFile, insightsApi, Params } from './api'
import { useData, useInsights } from './context'
import { ItemName } from './WorkspacePage'
import { Card, Empty, Meter, Seg, TextButton, fmtDay, fmtMoney, fmtTokens, fmtCount } from './ui'

const DIMS = ['purpose', 'operation', 'workspace', 'model', 'source', 'document', 'client'] as const
const NEXT: Record<string, string> = { purpose: 'operation', operation: 'workspace', workspace: 'purpose', model: 'operation', source: 'purpose' }

export default function ExploreTab() {
  const { t } = useTranslation()
  const { scope, set, params, explore, setExplore } = useInsights()
  const { drill } = explore
  const used = (dim: string) => (dim === 'purpose' && scope.purpose) || (dim === 'workspace' && scope.workspace) || drill.some((c) => c.dim === dim)
  const group = used(explore.group) ? DIMS.find((d) => !used(d)) ?? 'document' : explore.group
  const filters: Params = { ...params, purpose: scope.purpose, workspace: scope.workspace }
  for (const c of drill) filters[c.dim] = c.key
  const { data, error, loading } = useData(() => insightsApi.breakdown({ ...filters, group }), [JSON.stringify(filters), group])
  const leaf = group === 'document' || group === 'client'
  const selected = drill.find((c) => c.dim === 'call')
  const { data: calls } = useData(
    () => (selected ? insightsApi.calls({ ...filters, call: undefined, ...(selected.key.startsWith('client:') ? { client: selected.key.slice(7) } : { doc_id: selected.key }), day: filters.day as string, days: Math.min(Number(params.days) || 30, 90) }) : Promise.resolve({ rows: [] })),
    [selected?.key, JSON.stringify(filters)]
  )
  const label = (key: string) => {
    if (group === 'purpose') return t(`insights.purpose.${key}`, { defaultValue: key })
    if (group === 'operation') return t(`insights.operation.${key}`, { defaultValue: key })
    if (group === 'source') return t(`insights.source.${key}`, { defaultValue: key })
    if (group === 'model') return key || t('insights.models.unknown')
    return key
  }
  const rows: any[] = data?.rows ?? []
  const max = rows[0]?.tokens || 1
  const removable = (i: number) => {
    const rest = drill.slice(0, i)
    setExplore({ drill: rest, group: drill[i].dim === 'call' ? group : drill[i].dim === 'day' ? group : drill[i].dim })
  }
  const pick = (row: any) => {
    if (leaf) {
      const key = group === 'client' ? `client:${row.key}` : row.key
      setExplore({ drill: [...drill.filter((c) => c.dim !== 'call'), { dim: 'call', key, label: t('insights.explore.calls') }] })
      return
    }
    if (group === 'purpose') {
      set({ purpose: row.key })
      setExplore({ group: 'operation' })
      return
    }
    const next = [...drill, { dim: group, key: row.key, label: label(row.key) }]
    const after = NEXT[group]
    setExplore({ drill: next, group: !after || next.some((c) => c.dim === after) ? DIMS.find((d) => !next.some((c) => c.dim === d)) ?? 'document' : after })
  }
  return (
    <Card className={loading ? 'opacity-60 transition-opacity' : 'transition-opacity'}>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2.5">
        <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.explore.groupBy')}
          <Seg value={group} onChange={(g) => setExplore({ group: g, drill: drill.filter((c) => c.dim !== 'call') })} options={DIMS.map((d) => ({ value: d, label: t(`insights.explore.dim.${d}`), disabled: Boolean(used(d)) }))} />
        </div>
        <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => downloadFile('/breakdown.csv', `usage-by-${group}.csv`, { ...filters, group })}>{t('insights.explore.export')}</button>
      </div>
      <div className="mb-2.5 flex min-h-8 flex-wrap items-center gap-1.5 text-[13px]">
        <span className="text-muted-foreground text-xs">{t('insights.explore.path')}</span>
        <span className="rounded-full border px-2.5 py-0.5">{scope.workspace || t('insights.allWorkspaces')}</span>
        {scope.purpose ? <><span className="text-muted-foreground">›</span><button type="button" className="hover:bg-muted rounded-full border px-2.5 py-0.5" onClick={() => set({ purpose: '' })}>{t(`insights.purpose.${scope.purpose}`)} ×</button></> : null}
        {drill.map((c, i) => (
          <span key={`${c.dim}-${c.key}`} className="inline-flex items-center gap-1.5">
            <span className="text-muted-foreground">›</span>
            <button type="button" className="hover:bg-muted rounded-full border px-2.5 py-0.5" onClick={() => removable(i)} aria-label={t('insights.explore.remove', { name: c.label })}>{c.dim === 'day' ? fmtDay(c.key) : c.label} ×</button>
          </span>
        ))}
      </div>
      {error ? <Empty>{error}</Empty> : null}
      {selected ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[700px] text-[13px]">
            <thead><tr className="text-muted-foreground border-b text-xs">{['time', 'purpose', 'operation', 'model', 'prompt', 'answer', 'latency', 'status'].map((c) => <th key={c} className={`p-2 font-medium ${c === 'prompt' || c === 'answer' || c === 'latency' ? 'text-right' : 'text-left'}`}>{t(`insights.explore.call.${c}`)}</th>)}</tr></thead>
            <tbody>
              {(calls?.rows ?? []).map((r: any, i: number) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="p-2 whitespace-nowrap">{new Date(r.ts).toLocaleString()}</td>
                  <td className="p-2">{t(`insights.purpose.${r.purpose}`)}</td>
                  <td className="p-2">{t(`insights.operation.${r.operation}`, { defaultValue: r.operation })}</td>
                  <td className="p-2 font-mono text-xs">{r.model}{r.billing === 'byo' ? ' (BYO)' : ''}</td>
                  <td className="p-2 text-right">{fmtTokens(r.prompt_tokens)}</td>
                  <td className="p-2 text-right">{fmtTokens(r.completion_tokens)}</td>
                  <td className="p-2 text-right">{(r.latency_ms / 1000).toFixed(1)} s</td>
                  <td className="p-2">{r.cache_hit ? t('insights.explore.cacheHit') : t(`insights.explore.status.${r.status}`, { defaultValue: r.status })}{r.estimated ? ` · ${t('insights.explore.estimated')}` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!calls?.rows?.length ? <Empty>{t('insights.explore.noCalls')}</Empty> : null}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[600px] text-[13px]">
            <thead><tr className="text-muted-foreground border-b text-xs">
              <th className="p-2 text-left font-medium">{t(`insights.explore.dim.${group}`)}</th>
              <th className="w-[28%] p-2 text-left font-medium">{t('insights.explore.share')}</th>
              <th className="p-2 text-right font-medium">{t('insights.tokens')}</th>
              <th className="p-2 text-right font-medium">{t('insights.explore.callsCol')}</th>
              <th className="p-2 text-right font-medium">{t('insights.explore.perCall')}</th>
              <th className="p-2 text-right font-medium">{t('insights.estCost')}</th>
            </tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key} className="border-b last:border-0">
                  <td className="p-1.5"><TextButton className="break-all" onClick={() => pick(r)}>{r.display ? <ItemName display={r.display} /> : label(r.key)}</TextButton></td>
                  <td className="p-2"><Meter value={(r.tokens / max) * 100} /></td>
                  <td className="p-2 text-right">{fmtTokens(r.tokens)}</td>
                  <td className="p-2 text-right">{fmtCount(r.calls)}</td>
                  <td className="p-2 text-right">{r.calls ? fmtTokens(r.tokens / r.calls) : '–'}</td>
                  <td className="p-2 text-right">{fmtMoney(r.cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && !loading ? <Empty>{t('insights.empty')}</Empty> : null}
        </div>
      )}
      <p className="text-muted-foreground mt-3 mb-0 text-xs">
        {group === 'document' ? `${t('insights.explore.maskNote')} ` : ''}
        {leaf ? t('insights.explore.leafNote') : t('insights.explore.note')}
      </p>
    </Card>
  )
}

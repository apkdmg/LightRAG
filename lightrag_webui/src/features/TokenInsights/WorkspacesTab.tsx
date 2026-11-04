import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import Input from '@/components/ui/Input'
import { useDebounce } from '@/hooks/useDebounce'
import { insightsApi } from './api'
import { useData, useInsights } from './context'
import { Card, Empty, Seg, TextButton, Title } from '@/components/admin'
import { Mix, StatusLabel, fmtChange, fmtCount, fmtMoney, fmtPct, fmtTokens } from './ui'

type Sort = 'cost' | 'tokens' | 'change' | 'documents' | 'questions'

const SEARCH_DEBOUNCE_MS = 250

export default function WorkspacesTab() {
  const { t } = useTranslation()
  const { scope, set, params } = useInsights()
  const [sort, setSort] = useState<Sort>('cost')
  const [show, setShow] = useState<'all' | 'byo' | 'look'>('all')
  const [query, setQuery] = useState('')
  const q = useDebounce(query, SEARCH_DEBOUNCE_MS).trim()
  // A page belongs to one view, so changing the search or a filter goes back to the first.
  const view = JSON.stringify([params, scope.purpose, q, show, sort])
  const [paging, setPaging] = useState({ view, page: 1 })
  const page = paging.view === view ? paging.page : 1
  const { data, error, loading } = useData(
    () => insightsApi.workspaces({ ...params, purpose: scope.purpose, q, show, sort, page }),
    [view, page]
  )
  const rows: any[] = data?.rows ?? []
  const shownPage: number = data?.page ?? page
  const lastPage = data ? Math.max(Math.ceil(data.total / data.page_size), 1) : 1
  if (error && !data) return <Empty>{error}</Empty>
  return (
    <Card className={loading ? 'opacity-60 transition-opacity' : 'transition-opacity'}>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-x-4 gap-y-2.5">
        <Title sub={data ? t('insights.workspaces.sentence', { count: data.counts.workspaces, byo: data.counts.byo, look: data.counts.look }) : undefined}>{t('insights.workspaces.title')}</Title>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.workspaces.find')}
            <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t('insights.workspaces.placeholder')} className="h-9 w-44" />
          </label>
          <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.workspaces.show')}
            <Seg value={show} onChange={setShow} options={[{ value: 'all', label: t('insights.all') }, { value: 'byo', label: t('insights.workspaces.byo') }, { value: 'look', label: t('insights.workspaces.look') }]} />
          </div>
          <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.workspaces.sortBy')}
            <Seg value={sort} onChange={setSort} options={(['cost', 'tokens', 'change', 'documents', 'questions'] as Sort[]).map((value) => ({ value, label: t(`insights.workspaces.sort.${value}`) }))} />
          </div>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1040px] text-[13px]">
          <thead>
            <tr className="text-muted-foreground border-b text-xs">
              <th className="p-2 text-left font-medium">{t('insights.workspaces.workspace')}</th>
              <th className="p-2 text-right font-medium">{t('insights.estCost')}</th>
              <th className="p-2 text-right font-medium">{t('insights.workspaces.share')}</th>
              <th className="p-2 text-right font-medium">{t('insights.tokens')}</th>
              <th className="p-2 text-right font-medium">{t('insights.workspaces.sort.change')}</th>
              <th className="w-[12%] p-2 text-left font-medium">{t('insights.workspaces.mix')}</th>
              <th className="p-2 text-right font-medium">{t('insights.workspaces.sort.documents')}</th>
              <th className="p-2 text-right font-medium">{t('insights.workspaces.sort.questions')}</th>
              <th className="p-2 text-left font-medium">{t('insights.workspaces.whoPays')}</th>
              <th className="p-2 text-right font-medium">{t('insights.workspaces.lastActive')}</th>
              <th className="p-2 text-left font-medium">{t('insights.workspaces.status')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className={`border-b last:border-0 ${scope.workspace === r.id ? 'bg-muted' : ''}`}>
                <td className="p-2">
                  {r.deleted ? <span>{t('insights.workspaces.deleted', { code: r.id.replace('deleted-', '').slice(0, 4) })}</span> : <TextButton className="break-all" onClick={() => set({ workspace: r.id, purpose: '', tab: 'overview' })}>{r.id}</TextButton>}
                  {r.owner ? <div className="text-muted-foreground text-xs break-all">{r.owner}</div> : null}
                </td>
                <td className="p-2 text-right font-semibold">{fmtMoney(r.cost)}</td>
                <td className="p-2 text-right">{fmtPct(r.share_cost)}</td>
                <td className="p-2 text-right">{fmtTokens(r.tokens)}</td>
                <td className="p-2 text-right">{fmtChange(r.change)}</td>
                <td className="p-2"><Mix values={r.mix} /></td>
                <td className="p-2 text-right">{fmtCount(r.documents)}</td>
                <td className="p-2 text-right">{fmtCount(r.questions)}</td>
                <td className="p-2">{r.deleted ? '–' : r.byo_for.length ? t('insights.ranking.byoFor', { names: r.byo_for.map((p: string) => t(`insights.purpose.${p}`)).join(', ') }) : t('insights.workspacePage.systemKey')}</td>
                <td className="p-2 text-right whitespace-nowrap">{r.last_active ? new Date(r.last_active).toLocaleDateString() : '–'}</td>
                <td className="p-2">
                  <StatusLabel status={r.status} />
                  {r.open_anomalies || r.failed_documents ? <div className="text-muted-foreground text-xs">{[r.open_anomalies ? t('insights.workspaces.open', { count: r.open_anomalies }) : '', r.failed_documents ? t('insights.workspaces.failed', { count: r.failed_documents }) : ''].filter(Boolean).join(' · ')}</div> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {data && !rows.length ? <p className="text-muted-foreground mt-3 mb-0 text-xs">{t('insights.workspaces.none')}</p> : null}
      </div>
      {lastPage > 1 ? (
        <div className="mt-3 flex items-center justify-end gap-2 text-xs">
          <span className="text-muted-foreground me-auto">{t('insights.workspaces.matching', { count: data.total })}</span>
          <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px] disabled:opacity-50" disabled={shownPage <= 1} onClick={() => setPaging({ view, page: shownPage - 1 })}>{t('insights.workspaces.previousPage')}</button>
          <span className="text-muted-foreground">{t('insights.workspaces.pageOf', { page: shownPage, total: lastPage })}</span>
          <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px] disabled:opacity-50" disabled={shownPage >= lastPage} onClick={() => setPaging({ view, page: shownPage + 1 })}>{t('insights.workspaces.nextPage')}</button>
        </div>
      ) : null}
      <p className="text-muted-foreground mt-3 mb-0 text-xs">{t('insights.workspaces.deletedNote')}</p>
    </Card>
  )
}

import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowDownIcon, ArrowUpIcon } from 'lucide-react'
import Input from '@/components/ui/Input'
import { cn } from '@/lib/utils'
import { Seg } from '@/features/TokenInsights/ui'
import { useDebounce } from '@/hooks/useDebounce'
import { opsApi } from '../api'
import OpsLink from '../components/OpsLink'
import { Pager, Panel } from '../components/Panel'
import { StatusChip } from '../components/StatusChip'
import { useOpsNav } from '../context'
import { ageSeconds, fmtAge, fmtCount, utcTitle } from '../format'
import { useOpsPoll } from '../poll'
import { workspaceRowState } from '../severity'
import { BACKLOG_FILTERS, BACKLOG_SORTS, BacklogState, MAX_QUERY_LENGTH, backlogQuery, nextSort, toggleFilter } from './backlog'

const SEARCH_DEBOUNCE_MS = 250
/** Q2 pages are cached for 10 s on the server. */
const BACKLOG_SECONDS = 15
/** The age at which the oldest in-flight doc turns Warning / Serious (§1.7 oldest_pending). */
const OLDEST_WARN_S = 2 * 3600
const OLDEST_SERIOUS_S = 6 * 3600

const ageState = (s: number | null) => (s === null ? 'unknown' : s > OLDEST_SERIOUS_S ? 'serious' : s > OLDEST_WARN_S ? 'warning' : 'ok')

/**
 * Backlog by workspace (§2.4, FR-12): server-side search (name, owner or
 * owner email, debounced 250 ms), filters, sort and paging (50 per page), so
 * 3000 workspaces never reach the browser at once. Rows open the workspace
 * drawer.
 */
export default function BacklogView({ state, onChange }: { state: BacklogState; onChange: (next: BacklogState) => void }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const [query, setQuery] = useState(state.q)
  const q = useDebounce(query, SEARCH_DEBOUNCE_MS).trim()
  // A page belongs to one view: a new search, filter or sort goes back to the first page.
  const view = JSON.stringify([q, state.has, state.sort, state.dir])
  const [paging, setPaging] = useState({ view, page: state.page })
  const page = paging.view === view ? paging.page : 1
  const params = backlogQuery({ ...state, q, page })
  const poll = useOpsPoll((signal) => opsApi.workspaces(params, signal), [params], BACKLOG_SECONDS)
  const col = (key: string) => t(`operations.backlog.col.${key}`)

  return (
    <Panel
      poll={poll}
      freshness={BACKLOG_SECONDS}
      title={t('operations.backlog.byWorkspace')}
      sub={poll.data ? t('operations.backlog.matching', { count: poll.data.total, total: fmtCount(poll.data.total) }) : undefined}
    >
      {(data) => (
        <div className={cn('flex flex-col gap-3', poll.loading && 'opacity-60')}>
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-muted-foreground flex flex-col gap-0.5 text-xs">
              {t('operations.backlog.find')}
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t('operations.backlog.placeholder')}
                className="h-9 w-56"
                maxLength={MAX_QUERY_LENGTH}
              />
            </label>
            <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">
              {t('operations.backlog.filters')}
              <div className="flex max-w-full gap-1.5 overflow-x-auto" role="group" aria-label={t('operations.backlog.filters')}>
                {BACKLOG_FILTERS.map((f) => (
                  <button
                    key={f}
                    type="button"
                    aria-pressed={state.has.includes(f)}
                    onClick={() => onChange({ ...state, has: toggleFilter(state.has, f), page: 1 })}
                    className={cn(
                      'min-h-8 rounded-full border px-3 text-[13px] whitespace-nowrap transition-colors',
                      state.has.includes(f) ? 'bg-foreground text-background' : 'hover:bg-muted text-foreground'
                    )}
                  >
                    {t(`operations.backlog.has.${f}`)}
                  </button>
                ))}
              </div>
            </div>
            <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">
              {t('operations.backlog.sortBy')}
              <div className="flex items-center gap-1.5">
                <Seg
                  value={state.sort}
                  onChange={(s) => onChange({ ...state, ...nextSort(state, s), page: 1 })}
                  options={BACKLOG_SORTS.map((s) => ({ value: s, label: t(`operations.backlog.sort.${s}`) }))}
                />
                <button
                  type="button"
                  className="hover:bg-muted text-foreground inline-flex min-h-8 items-center gap-1 rounded-lg border px-2 text-[13px]"
                  onClick={() => onChange({ ...state, ...nextSort(state, state.sort), page: 1 })}
                  aria-label={t(state.dir === 'asc' ? 'operations.backlog.dirAsc' : 'operations.backlog.dirDesc')}
                  title={t(state.dir === 'asc' ? 'operations.backlog.dirAsc' : 'operations.backlog.dirDesc')}
                >
                  {state.dir === 'asc' ? <ArrowUpIcon className="size-3.5" aria-hidden="true" /> : <ArrowDownIcon className="size-3.5" aria-hidden="true" />}
                </button>
              </div>
            </div>
          </div>
          {data.items.length ? (
            <div className="overflow-x-auto">
              <table className="ops-stack w-full min-w-[980px] text-[13px]">
                <thead>
                  <tr className="text-muted-foreground border-b text-xs">
                    <th className="p-2 text-left font-medium">{col('workspace')}</th>
                    {(['pending', 'parsing', 'analyzing', 'processing', 'failed', 'cancelled'] as const).map((k) => (
                      <th key={k} className="p-2 text-right font-medium">
                        {col(k)}
                      </th>
                    ))}
                    <th className="p-2 text-left font-medium">{col('oldest')}</th>
                    <th className="p-2 text-right font-medium">{col('queue')}</th>
                    <th className="p-2 text-left font-medium">{col('state')}</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((r) => {
                    const oldest = r.oldest_inflight_at ? ageSeconds(r.oldest_inflight_at, poll.updatedAt ?? undefined) : null
                    const to = { ...nav.view, drawer: { kind: 'workspace' as const, id: r.workspace } }
                    const open = () => nav.openDrawer({ kind: 'workspace', id: r.workspace })
                    return (
                      <tr
                        key={r.workspace}
                        className="ops-row-link border-b align-top last:border-0"
                        onClick={(e) => {
                          if ((e.target as HTMLElement).closest('a')) return
                          open()
                        }}
                      >
                        <td className="p-2" data-label={col('workspace')}>
                          <span className="flex flex-col items-end md:items-start">
                            <OpsLink to={to} onNavigate={open} className="break-all underline-offset-2 hover:underline">
                              {r.workspace}
                            </OpsLink>
                            {r.owner ? <span className="text-muted-foreground text-xs break-all">{r.owner}</span> : null}
                          </span>
                        </td>
                        {(['pending', 'parsing', 'analyzing', 'processing', 'failed', 'cancelled'] as const).map((k) => (
                          <td key={k} className="p-2 text-right tabular-nums" data-label={col(k)}>
                            {fmtCount(r[k] ?? null)}
                          </td>
                        ))}
                        <td className="p-2" data-label={col('oldest')} title={utcTitle(r.oldest_inflight_at)}>
                          {oldest === null ? '–' : <StatusChip state={ageState(oldest)}>{fmtAge(oldest)}</StatusChip>}
                        </td>
                        <td className="p-2 text-right tabular-nums" data-label={col('queue')}>
                          {fmtCount(r.queue_pending ?? null)}
                        </td>
                        <td className="p-2" data-label={col('state')}>
                          {r.paused ? (
                            <StatusChip state="paused">{t('operations.backlog.pausedState')}</StatusChip>
                          ) : r.state ? (
                            <StatusChip state={workspaceRowState(r.state)}>{t(`operations.backlog.state.${r.state}`, { defaultValue: r.state })}</StatusChip>
                          ) : (
                            '–'
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-muted-foreground m-0 text-sm">{q || state.has.length ? t('operations.backlog.noneMatching') : t('operations.backlog.none')}</p>
          )}
          <Pager page={data.page} pageSize={data.page_size} total={data.total} onPage={(p) => setPaging({ view, page: p })} />
        </div>
      )}
    </Panel>
  )
}

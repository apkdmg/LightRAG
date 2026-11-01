import { useCallback, useEffect, useMemo, useState } from 'react'
import { AnimatePresence } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { CheckIcon, ChevronsUpDownIcon, SettingsIcon } from 'lucide-react'
import { Command, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/Command'
import Input from '@/components/ui/Input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { useDebounce } from '@/hooks/useDebounce'
import AnomaliesTab from './AnomaliesTab'
import { Params, insightsApi } from './api'
import { Ctx, Drill, Scope, useData } from './context'
import ExploreTab from './ExploreTab'
import ForecastTab from './ForecastTab'
import { ICON_CHECK, ICON_CLOCK, PURPOSES, Purpose, SEVERITY } from './meta'
import OverviewTab from './OverviewTab'
import SettingsDialog from './SettingsDialog'
import { Card, Empty, Seg, StatusIcon, Swatch, ViewBoundary, fmtDay } from './ui'
import WhyPanel from './WhyPanel'
import WorkspacesTab from './WorkspacesTab'
import WorkTab from './WorkTab'
import './insights.css'

const TABS = ['overview', 'workspaces', 'explore', 'anomalies', 'work', 'forecast']
const SEARCH_DEBOUNCE_MS = 250
const ALL_OPTION = '__all__'
const DEFAULT: Scope = { tab: 'overview', purpose: '', workspace: '', range: 30, start: '', end: '', unit: 'tokens', day: null }

/** The URL holds the view, so an alert email or a colleague can open it directly. */
const readHash = (): Partial<Scope> => {
  const query = window.location.hash.split('?')[1]
  if (!query) return {}
  const q = new URLSearchParams(query)
  if (q.get('insights') !== '1') return {}
  const out: Partial<Scope> = {}
  const tab = q.get('itab')
  if (tab && TABS.includes(tab)) out.tab = tab
  const purpose = q.get('ipurpose')
  if (purpose && (PURPOSES as readonly string[]).includes(purpose)) out.purpose = purpose as Purpose
  if (q.get('iws')) out.workspace = q.get('iws') as string
  if (q.get('iday')) out.day = q.get('iday')
  const range = q.get('irange')
  if (range === 'custom' && q.get('istart') && q.get('iend')) {
    out.range = 'custom'
    out.start = q.get('istart') as string
    out.end = q.get('iend') as string
  } else if (range && !Number.isNaN(Number(range))) {
    out.range = Number(range)
  }
  return out
}

const writeHash = (scope: Scope) => {
  const q = new URLSearchParams({ insights: '1', itab: scope.tab, irange: String(scope.range) })
  if (scope.purpose) q.set('ipurpose', scope.purpose)
  if (scope.workspace) q.set('iws', scope.workspace)
  if (scope.day) q.set('iday', scope.day)
  if (scope.range === 'custom') {
    q.set('istart', scope.start)
    q.set('iend', scope.end)
  }
  window.history.replaceState(null, '', `#/?${q.toString()}`)
}

/** Thousands of workspaces: nothing loads until the list opens, and the server searches. */
function WorkspacePicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const q = useDebounce(search, SEARCH_DEBOUNCE_MS).trim()
  const { data, error, loading } = useData<any>(
    () => (open ? insightsApi.workspaceOptions({ q }) : Promise.resolve(null)),
    [open, q]
  )
  const handleOpenChange = (next: boolean) => {
    setOpen(next)
    if (!next) setSearch('')
  }
  const pick = (workspace: string) => {
    onChange(workspace)
    handleOpenChange(false)
  }
  const options: { id: string; owner: string | null }[] = data?.options ?? []
  // New results replace the list, so Enter would otherwise have nothing to pick.
  const [moved, setMoved] = useState<{ results: unknown; value: string } | null>(null)
  const highlighted = moved && moved.results === data ? moved.value : q ? options[0]?.id ?? '' : ALL_OPTION
  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <button type="button" role="combobox" aria-expanded={open} aria-label={t('insights.control.workspace')} className="bg-card flex min-h-[38px] w-[200px] items-center justify-between gap-2 rounded-lg border px-2.5 text-sm">
          <span className="truncate">{value || t('insights.allWorkspaces')}</span>
          <ChevronsUpDownIcon className="size-3.5 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-[260px] p-0" align="start">
        <Command shouldFilter={false} value={highlighted} onValueChange={(next) => setMoved({ results: data, value: next })}>
          <CommandInput value={search} onValueChange={setSearch} placeholder={t('insights.workspaces.placeholder')} />
          <CommandList className={loading ? 'opacity-60' : ''}>
            {error ? <p className="text-muted-foreground px-3 py-3 text-xs">{error}</p> : null}
            {data && !options.length && q ? <p className="text-muted-foreground px-3 py-3 text-center text-sm">{t('insights.workspaces.none')}</p> : null}
            <CommandGroup>
              {!search.trim() ? (
                <CommandItem value={ALL_OPTION} onSelect={() => pick('')}>
                  <CheckIcon className={`me-1.5 size-3.5 ${value === '' ? 'opacity-100' : 'opacity-0'}`} />
                  <span className="truncate">{t('insights.allWorkspaces')}</span>
                </CommandItem>
              ) : null}
              {options.map((o) => (
                <CommandItem key={o.id} value={o.id} onSelect={() => pick(o.id)}>
                  <CheckIcon className={`me-1.5 size-3.5 shrink-0 ${value === o.id ? 'opacity-100' : 'opacity-0'}`} />
                  <span className="min-w-0">
                    <span className="block truncate">{o.id}</span>
                    {o.owner ? <span className="text-muted-foreground block truncate text-xs">{o.owner}</span> : null}
                  </span>
                </CommandItem>
              ))}
            </CommandGroup>
            {data && data.total > options.length ? <p className="text-muted-foreground border-t px-3 py-2 text-xs">{t('insights.workspaces.showingOf', { shown: options.length, total: data.total })}</p> : null}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

function Attention({ overview, scope, set }: { overview: any; scope: Scope; set: (p: Partial<Scope>) => void }) {
  const { t } = useTranslation()
  if (!overview) return null
  const { attention, learning, budget } = overview
  if (!learning.anomalies_ready) {
    return (
      <Card className="flex flex-wrap items-center gap-x-3.5 gap-y-2.5 px-4 py-3">
        <StatusIcon path={ICON_CLOCK} color="var(--ti-axis)" />
        <span className="flex-[1_1_340px] text-[15px] font-medium">
          {learning.recording_since
            ? t('insights.attention.learning', { since: fmtDay(learning.recording_since), anomalies: learning.anomalies_in_days, forecast: learning.forecast_in_days })
            : t('insights.attention.noData')}
        </span>
      </Card>
    )
  }
  const names = attention.above.map((p: string) => t(`insights.purpose.${p}`))
  const joined = names.length > 1 ? `${names.slice(0, -1).join(', ')} ${t('insights.and')} ${names[names.length - 1]}` : names[0]
  const now = names.length
    ? t(names.length > 1 ? 'insights.attention.abovePlural' : 'insights.attention.above', { names: joined }) + (attention.reason ? ` ${t('insights.attention.because', { source: t(`insights.source.${attention.reason.source}`, { defaultValue: attention.reason.source }), workspace: attention.reason.workspace })}` : '')
    : t('insights.attention.normal')
  const docs = attention.documents
  const stuck = docs && (docs.failed || docs.pending) ? t('insights.attention.documents', { waiting: docs.pending || 0, failed: docs.failed || 0 }) : ''
  const open = attention.open ? t(attention.critical ? 'insights.attention.critical' : 'insights.attention.open', { count: attention.critical || attention.open }) : ''
  const money = !scope.workspace && budget?.set && budget.runs_out ? t('insights.attention.budget', { day: fmtDay(budget.runs_out) }) : ''
  const look = attention.critical ? SEVERITY.critical : attention.above.length || docs?.failed ? SEVERITY.warning : { path: ICON_CHECK, color: '#0ca30c' }
  return (
    <Card className="flex flex-wrap items-center gap-x-3.5 gap-y-2.5 px-4 py-3">
      <StatusIcon path={look.path} color={look.color} />
      <span className="flex-[1_1_340px] text-[15px] font-medium">{[scope.workspace ? `${scope.workspace}:` : '', now, stuck, open, money].filter(Boolean).join(' ')}</span>
      {attention.open ? <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => set({ tab: 'anomalies', purpose: '' })}>{t('insights.attention.review')}</button> : null}
      {!scope.workspace ? <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => set({ tab: 'forecast' })}>{t('insights.attention.forecast')}</button> : null}
    </Card>
  )
}

export default function TokenInsights() {
  return <ViewBoundary><InsightsPage /></ViewBoundary>
}

function InsightsPage() {
  const { t } = useTranslation()
  const [scope, setScope] = useState<Scope>(() => ({ ...DEFAULT, ...readHash() }))
  const [explore, setExploreState] = useState<{ group: string; drill: Drill[] }>({ group: 'purpose', drill: [] })
  const [settingsOpen, setSettingsOpen] = useState(false)
  const set = useCallback((patch: Partial<Scope>) => setScope((s) => ({ ...s, ...patch })), [])
  const setExplore = useCallback((patch: { group?: string; drill?: Drill[] }) => setExploreState((e) => ({ ...e, ...patch })), [])
  const goExplore = useCallback((group: string, drill: Drill[], patch?: Partial<Scope>) => {
    setExploreState({ group, drill })
    setScope((s) => ({ ...s, tab: 'explore', ...patch }))
  }, [])
  useEffect(() => writeHash(scope), [scope])
  // Leaving the tab drops the view from the URL, so a reload stays where the user went.
  useEffect(() => () => window.history.replaceState(null, '', '#/'), [])

  const params: Params = useMemo(
    () => (scope.range === 'custom' && scope.start && scope.end ? { start: scope.start, end: scope.end } : { days: scope.range === 'custom' ? 30 : scope.range }),
    [scope.range, scope.start, scope.end]
  )
  const status = useData(() => insightsApi.status(), [])
  const head = useData(
    () => (status.data?.available ? insightsApi.overview({ ...params, workspace: scope.workspace }) : Promise.resolve(null)),
    [status.data?.available, JSON.stringify(params), scope.workspace],
    60
  )
  const models = useMemo<string[]>(() => (head.data?.models ?? []).map((m: any) => m.id), [head.data])
  const refreshHeader = head.reload
  const contextValue = useMemo(() => ({ scope, set, params, explore, setExplore, goExplore, refreshHeader }), [scope, set, params, explore, setExplore, goExplore, refreshHeader])

  if (status.error) return <div className="ti-root p-6"><Empty>{status.error}</Empty></div>
  if (status.data && !status.data.available) {
    return (
      <div className="ti-root p-6">
        <Card className="max-w-[640px]"><h2 className="m-0 mb-1.5 text-lg font-semibold">{t('insights.unavailable.title')}</h2><p className="text-muted-foreground m-0 text-sm">{t('insights.unavailable.body')}</p></Card>
      </div>
    )
  }
  const quality = head.data?.quality
  const estimated = quality?.calls ? Math.round((quality.estimated_calls / quality.calls) * 1000) / 10 : 0
  return (
    <Ctx.Provider value={contextValue}>
      <div className="ti-root relative h-full overflow-hidden">
        <div className="h-full overflow-y-auto">
          <div className="mx-auto flex max-w-[1360px] flex-col gap-4 px-4 pt-5 pb-14 md:px-7">
            <div className="flex flex-wrap items-end justify-between gap-2.5">
              <div>
                <h1 className="m-0 text-[26px] font-semibold tracking-tight">{t('insights.title')}</h1>
                <div className="text-muted-foreground mt-0.5 text-[13px]">
                  {head.data?.learning?.recording_since ? t('insights.freshness', { since: fmtDay(head.data.learning.recording_since), exact: (100 - estimated).toFixed(1), estimated: estimated.toFixed(1) }) : t('insights.noRecordingYet')}
                  {quality?.lost_rows ? ` · ${t('insights.lostRows', { count: quality.lost_rows })}` : ''}
                </div>
              </div>
              <button type="button" className="hover:bg-muted inline-flex min-h-9 items-center gap-1.5 rounded-lg border px-3 text-[13px]" onClick={() => setSettingsOpen(true)}><SettingsIcon className="size-3.5" />{t('insights.settings.open')}</button>
            </div>
            <Attention overview={head.data} scope={scope} set={set} />
            <div className="flex flex-wrap items-end gap-x-3 gap-y-2.5">
              <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.control.purpose')}
                <Seg label={t('insights.control.purpose')} value={scope.purpose} onChange={(purpose) => set({ purpose })} options={[{ value: '' as Purpose | '', label: t('insights.all') }, ...PURPOSES.map((p) => ({ value: p as Purpose | '', label: <><Swatch purpose={p} />{t(`insights.purpose.${p}`)}</> }))]} />
              </div>
              <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.control.workspace')}
                <WorkspacePicker value={scope.workspace} onChange={(workspace) => { set({ workspace }); setExplore({ drill: explore.drill.filter((c) => c.dim !== 'workspace') }) }} />
              </div>
              <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.control.range')}
                <Seg label={t('insights.control.range')} value={scope.range} onChange={(range) => set(range === 'custom' ? { range, start: scope.start || new Date(Date.now() - 13 * 864e5).toISOString().slice(0, 10), end: scope.end || new Date().toISOString().slice(0, 10) } : { range })} options={[{ value: 1 as number | 'custom', label: t('insights.control.today') }, { value: 7, label: t('insights.control.days', { count: 7 }) }, { value: 30, label: t('insights.control.days', { count: 30 }) }, { value: 90, label: t('insights.control.days', { count: 90 }) }, { value: 'custom', label: t('insights.control.custom') }]} />
              </div>
              {scope.range === 'custom' ? (
                <div className="flex items-end gap-1.5">
                  <Input type="date" aria-label={t('insights.control.from')} value={scope.start} max={scope.end} onChange={(e) => set({ start: e.target.value })} className="h-[38px] w-[150px]" />
                  <Input type="date" aria-label={t('insights.control.to')} value={scope.end} min={scope.start} max={new Date().toISOString().slice(0, 10)} onChange={(e) => set({ end: e.target.value })} className="h-[38px] w-[150px]" />
                </div>
              ) : null}
              <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">{t('insights.control.showAs')}
                <Seg label={t('insights.control.showAs')} value={scope.unit} onChange={(unit) => set({ unit })} options={[{ value: 'tokens', label: t('insights.tokens') }, { value: 'cost', label: t('insights.estCost') }]} />
              </div>
              {scope.purpose || scope.workspace ? <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => { set({ purpose: '', workspace: '' }); setExplore({ drill: [], group: 'purpose' }) }}>{t('insights.control.clear')}</button> : null}
            </div>
            <div role="tablist" aria-label={t('insights.title')} className="flex gap-0.5 overflow-x-auto overflow-y-hidden border-b">
              {TABS.map((tab) => (
                <button key={tab} type="button" role="tab" aria-selected={scope.tab === tab} onClick={() => set({ tab })} className={`-mb-px border-b-2 px-3 py-2 text-sm whitespace-nowrap transition-colors ${scope.tab === tab ? 'border-foreground font-medium' : 'text-muted-foreground hover:text-foreground border-transparent'}`}>
                  {t(`insights.tab.${tab}`)}{tab === 'anomalies' && head.data?.attention?.open && head.data?.learning?.anomalies_ready ? ` (${head.data.attention.open})` : ''}
                </button>
              ))}
            </div>
            <ViewBoundary resetKey={JSON.stringify([scope.tab, scope.purpose, scope.workspace, params, scope.unit])}>
              {scope.tab === 'overview' ? <OverviewTab /> : null}
              {scope.tab === 'workspaces' ? <WorkspacesTab /> : null}
              {scope.tab === 'explore' ? <ExploreTab /> : null}
              {scope.tab === 'anomalies' ? <AnomaliesTab /> : null}
              {scope.tab === 'work' ? <WorkTab /> : null}
              {scope.tab === 'forecast' ? <ForecastTab onOpenSettings={() => setSettingsOpen(true)} /> : null}
            </ViewBoundary>
          </div>
        </div>
        <AnimatePresence>{scope.day ? <WhyPanel key="why" /> : null}</AnimatePresence>
        <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} onSaved={() => { head.reload(); status.reload() }} knownModels={models} backfill={head.data?.learning?.backfill} />
      </div>
    </Ctx.Provider>
  )
}

import { KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MotionConfig } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { PauseIcon, PlayIcon, RefreshCwIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { InsightsTarget, insightsLink } from '@/features/TokenInsights/hash'
import { Card, ViewBoundary } from '@/features/TokenInsights/ui'
import { actionsEnabled } from './actions'
import ActivityTab from './ActivityTab'
import { OpsOverview, OpsStatus, opsApi } from './api'
import DrawerStack from './components/DrawerStack'
import ErrorPanel from './components/ErrorPanel'
import OpsBanners from './components/OpsBanners'
import ProcessChip from './components/ProcessChip'
import { HostedActionDialogs } from './components/WorkspaceActions'
import RefreshHairline from './components/RefreshHairline'
import Skeleton from './components/Skeleton'
import { OverallStateChip } from './components/StatusChip'
import { DrawerMove, OpsNav, OpsNavContext, OpsOverviewContext, OpsShellContext, useOpsNav } from './context'
import CorrelateBar from './components/CorrelateBar'
import AlertDrawer from './drawers/AlertDrawer'
import DependencyDrawer from './drawers/DependencyDrawer'
import DocumentDrawer from './drawers/DocumentDrawer'
import LogEntryDrawer from './drawers/LogEntryDrawer'
import RunDrawer from './drawers/RunDrawer'
import WorkerDrawer from './drawers/WorkerDrawer'
import WorkspaceDrawer from './drawers/WorkspaceDrawer'
import { ageSeconds, fmtAge, fmtClock } from './format'
import { DEFAULT_OPS_HASH, Drawer, OPS_TABS, OpsHash, OpsTab, readOpsHash, sameDrawer, writeOpsHash } from './hashState'
import HealthTab from './HealthTab'
import { useNow, useStale } from './hooks'
import LogsTab from './LogsTab'
import { AttentionSlot, AttentionSlotContext, useOpsMotion } from './motion'
import { OVERVIEW_REFRESH_SECONDS, attentionCounts, overallState, overallWord, ringCandidates } from './overview'
import OverviewTab from './OverviewTab'
import PerformanceTab from './PerformanceTab'
import { OpsPoll, useOpsPoll } from './poll'
import { OpsPrefsContext, PAUSE_KEY, REDUCE_MOTION_KEY, readFlag, useOpsPrefs, writeFlag } from './prefs'
import ProcessingTab from './ProcessingTab'
import { isStale } from './severity'
import './ops.css'

/** /status changes rarely (mode, worker count); one minute keeps the header honest. */
const STATUS_REFRESH_SECONDS = 60

const PANELS: Record<OpsTab, () => React.JSX.Element> = {
  overview: OverviewTab,
  processing: ProcessingTab,
  logs: LogsTab,
  health: HealthTab,
  performance: PerformanceTab,
  activity: ActivityTab
}

export default function Operations() {
  return (
    <ViewBoundary>
      <OperationsPage />
    </ViewBoundary>
  )
}

function Freshness({ poll }: { poll: OpsPoll<OpsStatus> }) {
  const { t } = useTranslation()
  const now = useNow(15_000)
  if (!poll.updatedAt) return null
  const stale = isStale(poll.updatedAt, STATUS_REFRESH_SECONDS, now)
  return stale ? (
    <span className="rounded-md border border-dashed px-1.5 py-0.5 text-xs text-[color:var(--ops-stale-fg)]">
      {t('operations.stale', { age: fmtAge(ageSeconds(poll.updatedAt, now)) })}
    </span>
  ) : (
    <span className="text-muted-foreground font-mono text-xs">{t('operations.updated', { time: fmtClock(poll.updatedAt) })}</span>
  )
}

/** The overall state from /overview; the only aria-live region of the page besides the "N new" pill. */
function StateChip({ poll }: { poll: OpsPoll<OpsOverview> }) {
  const { stale } = useStale(poll.updatedAt, OVERVIEW_REFRESH_SECONDS)
  const data = poll.data
  if (!data) return poll.loading && !poll.error ? <Skeleton className="h-6 w-40" /> : null
  if (data.available === false) return null
  return (
    <OverallStateChip
      state={overallState(data)}
      word={overallWord(data.state, data.maintenance)}
      counts={attentionCounts(data.attention)}
      stale={stale || !!poll.error}
    />
  )
}

function Header({ poll, overview, reloadAll }: { poll: OpsPoll<OpsStatus>; overview: OpsPoll<OpsOverview>; reloadAll: () => void }) {
  const { t } = useTranslation()
  const { paused, reduced, osReduced } = useOpsMotion()
  // The button shows the effective state. While the OS asks for reduced motion
  // the page cannot switch it back on, so the button is locked on.
  const prefs = useOpsPrefs()
  const status = poll.data
  return (
    <div className="relative flex flex-wrap items-center gap-x-4 gap-y-2.5">
      <h1 className="m-0 text-[26px] font-semibold tracking-tight">{t('operations.title')}</h1>
      <div aria-live="polite" className="inline-flex">
        <StateChip poll={overview} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {status ? <ProcessChip process={status.process} /> : poll.loading ? <Skeleton className="h-6 w-36" /> : null}
        {status?.version ? <span className="text-muted-foreground text-xs">{t('operations.version', { version: status.version })}</span> : null}
      </div>
      <div className="ms-auto flex flex-wrap items-center gap-2">
        <Freshness poll={poll} />
        <button
          type="button"
          onClick={reloadAll}
          className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px]"
        >
          <RefreshCwIcon className="size-3.5" aria-hidden="true" />
          {t('operations.refresh')}
        </button>
        <button
          type="button"
          aria-pressed={paused}
          onClick={() => prefs.setPaused(!paused)}
          title={paused ? t('operations.resumeAutoHint') : t('operations.pauseAutoHint')}
          className={cn('hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px]', paused && 'bg-muted font-medium')}
        >
          {paused ? <PlayIcon className="size-3.5" aria-hidden="true" /> : <PauseIcon className="size-3.5" aria-hidden="true" />}
          {paused ? t('operations.resumeAuto') : t('operations.pauseAuto')}
        </button>
        <button
          type="button"
          aria-pressed={reduced}
          disabled={osReduced}
          title={osReduced ? t('operations.reduceMotionBySystem') : undefined}
          onClick={() => prefs.setReduceMotion(!prefs.reduceMotion)}
          className={cn('hover:bg-muted min-h-8 rounded-lg border px-2.5 text-[13px] disabled:cursor-not-allowed disabled:opacity-70', reduced && 'bg-muted font-medium')}
        >
          {t('operations.reduceMotion')}
        </button>
      </div>
      <div className="basis-full">
        <CorrelateBar />
      </div>
      <RefreshHairline slow={poll.slow || overview.slow} />
    </div>
  )
}

function TabRow({ tab, onSelect, badges }: { tab: OpsTab; onSelect: (tab: OpsTab) => void; badges?: Partial<Record<OpsTab, number>> }) {
  const { t } = useTranslation()
  const refs = useRef<Partial<Record<OpsTab, HTMLButtonElement | null>>>({})
  // Arrow keys, Home and End move between tabs (automatic activation).
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = OPS_TABS.indexOf(tab)
    let next: OpsTab | null = null
    if (e.key === 'ArrowRight') next = OPS_TABS[(i + 1) % OPS_TABS.length]
    else if (e.key === 'ArrowLeft') next = OPS_TABS[(i - 1 + OPS_TABS.length) % OPS_TABS.length]
    else if (e.key === 'Home') next = OPS_TABS[0]
    else if (e.key === 'End') next = OPS_TABS[OPS_TABS.length - 1]
    if (!next) return
    e.preventDefault()
    onSelect(next)
    refs.current[next]?.focus()
  }
  return (
    <div role="tablist" aria-label={t('operations.tabsLabel')} onKeyDown={onKeyDown} className="flex gap-0.5 overflow-x-auto overflow-y-hidden border-b">
      {OPS_TABS.map((id) => (
        <button
          key={id}
          ref={(el) => {
            refs.current[id] = el
          }}
          id={`ops-tab-${id}`}
          type="button"
          role="tab"
          aria-selected={tab === id}
          aria-controls={`ops-panel-${id}`}
          tabIndex={tab === id ? 0 : -1}
          onClick={() => onSelect(id)}
          className={cn(
            '-mb-px border-b-2 px-3 py-2 text-sm whitespace-nowrap transition-colors',
            tab === id ? 'border-foreground font-medium' : 'text-muted-foreground hover:text-foreground border-transparent'
          )}
        >
          {t(`operations.tab.${id}`)}
          {badges?.[id] ? (
            <span className="bg-muted ms-1.5 rounded-full px-1.5 py-px font-mono text-[11px]">
              <span aria-hidden="true">{badges[id]}</span>
              <span className="sr-only">{t('operations.tabBadge', { count: badges[id] })}</span>
            </span>
          ) : null}
        </button>
      ))}
    </div>
  )
}

type NavState = {
  view: OpsHash
  stack: Drawer[]
  move: DrawerMove
  /** False for a drawer that came with the page's URL: a deep link does not animate. */
  animate: boolean
}

function OperationsPage() {
  const [nav, setNav] = useState<NavState>(() => ({
    view: readOpsHash(window.location.hash) ?? DEFAULT_OPS_HASH,
    stack: [],
    move: 'none',
    animate: false
  }))
  // Only a tab the user switched to fades in; the first paint and deep links do not.
  const [fadeTab, setFadeTab] = useState<OpsTab | null>(null)
  const [paused, setPausedState] = useState(() => readFlag(PAUSE_KEY))
  const [reduceMotion, setReduceMotionState] = useState(() => readFlag(REDUCE_MOTION_KEY))

  const setPaused = useCallback((on: boolean) => {
    writeFlag(PAUSE_KEY, on)
    setPausedState(on)
  }, [])
  const setReduceMotion = useCallback((on: boolean) => {
    writeFlag(REDUCE_MOTION_KEY, on)
    setReduceMotionState(on)
  }, [])
  const prefs = useMemo(() => ({ paused, setPaused, reduceMotion, setReduceMotion }), [paused, setPaused, reduceMotion, setReduceMotion])

  const view = nav.view
  // The URL holds the view; leaving the tab drops it, so a reload lands where the user went.
  useEffect(() => {
    const next = writeOpsHash(view)
    if (window.location.hash !== next) window.history.replaceState(null, '', next)
  }, [view])
  useEffect(() => () => window.history.replaceState(null, '', '#/'), [])

  /** A view from a link or a pasted URL: the breadcrumb survives only if the drawer stays the same. */
  const setView = useCallback((next: OpsHash) => {
    setNav((s) => {
      if (writeOpsHash(next) === writeOpsHash(s.view)) return s
      const keep = sameDrawer(next.drawer, s.view.drawer)
      return {
        view: next,
        stack: keep ? s.stack : [],
        move: keep ? s.move : next.drawer ? (s.view.drawer ? 'push' : 'open') : s.view.drawer ? 'close' : s.move,
        animate: true
      }
    })
  }, [])

  // A pasted link while the tab is open switches the view too.
  useEffect(() => {
    const onHash = () => {
      const next = readOpsHash(window.location.hash)
      if (next) setView(next)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [setView])

  // Radix portals render outside .ops-root; the page toggle reaches them through <html>.
  useEffect(() => {
    const root = document.documentElement
    if (reduceMotion) root.setAttribute('data-ops-motion', 'reduced')
    else root.removeAttribute('data-ops-motion')
    return () => root.removeAttribute('data-ops-motion')
  }, [reduceMotion])

  const go = useCallback((tab: OpsTab, sub?: string | null) => {
    setNav((s) => {
      const sameTab = s.view.tab === tab
      const nextSub = sub === undefined ? (sameTab ? s.view.sub : null) : sub
      if (sameTab && nextSub === s.view.sub) return s
      // A tab switch closes the drawer; a sub-view switch keeps it.
      return {
        ...s,
        view: { ...s.view, tab, sub: nextSub, drawer: sameTab ? s.view.drawer : null, logQuery: sameTab ? s.view.logQuery : null },
        stack: sameTab ? s.stack : [],
        move: sameTab || !s.view.drawer ? s.move : 'close',
        animate: true
      }
    })
    setFadeTab((current) => (current === tab ? current : tab))
  }, [])
  const setSub = useCallback((sub: string) => setNav((s) => (s.view.sub === sub ? s : { ...s, view: { ...s.view, sub } })), [])
  const openDrawer = useCallback((drawer: Drawer, opts?: { push?: boolean }) => {
    setNav((s) => {
      if (sameDrawer(drawer, s.view.drawer)) return s
      const current = s.view.drawer
      return {
        view: { ...s.view, drawer },
        stack: opts?.push && current ? [...s.stack, current] : [],
        move: current ? 'push' : 'open',
        animate: true
      }
    })
  }, [])
  const backDrawer = useCallback((index?: number) => {
    setNav((s) => {
      if (!s.stack.length) return s
      const i = index === undefined ? s.stack.length - 1 : Math.max(0, Math.min(index, s.stack.length - 1))
      return { view: { ...s.view, drawer: s.stack[i] }, stack: s.stack.slice(0, i), move: 'back', animate: true }
    })
  }, [])
  const closeDrawer = useCallback(() => {
    setNav((s) => (s.view.drawer ? { ...s, view: { ...s.view, drawer: null }, stack: [], move: 'close', animate: true } : s))
  }, [])

  const navValue: OpsNav = useMemo(
    () => ({ view, stack: nav.stack, move: nav.move, go, setSub, openDrawer, backDrawer, closeDrawer, setView }),
    [view, nav.stack, nav.move, go, setSub, openDrawer, backDrawer, closeDrawer, setView]
  )

  const selectTab = useCallback((tab: OpsTab) => go(tab, null), [go])

  /**
   * Cost and token questions belong to Token Insights (§2.1). The Insights page
   * reads its hash while it renders, before this page's unmount clears the
   * URL, so a workspace, purpose or document link (`idoc`) lands where it points.
   */
  const openInsights = useCallback((target?: InsightsTarget) => {
    if (target && (target.workspace || target.docId || target.purpose || target.tab)) {
      window.history.replaceState(null, '', insightsLink(target))
    }
    useSettingsStore.getState().setCurrentTab('insights')
  }, [])

  return (
    <OpsPrefsContext.Provider value={prefs}>
      <OpsNavContext.Provider value={navValue}>
        {/* The first MotionConfig in the app, scoped to Operations: 'always' honours the page toggle. */}
        <MotionConfig reducedMotion={reduceMotion ? 'always' : 'user'}>
          <StatusShell fadeTab={fadeTab} selectTab={selectTab} openInsights={openInsights} animateDrawer={nav.animate} />
        </MotionConfig>
      </OpsNavContext.Provider>
    </OpsPrefsContext.Provider>
  )
}

function renderDrawer(drawer: Drawer) {
  if (drawer.kind === 'workspace') return <WorkspaceDrawer workspace={drawer.id} />
  if (drawer.kind === 'document') return <DocumentDrawer workspace={drawer.workspace} docId={drawer.id} />
  if (drawer.kind === 'run') return <RunDrawer runId={drawer.id} />
  if (drawer.kind === 'dependency') return <DependencyDrawer dep={drawer.id} />
  if (drawer.kind === 'worker') return <WorkerDrawer workerId={drawer.id} />
  if (drawer.kind === 'log') return <LogEntryDrawer id={drawer.id} />
  return <AlertDrawer alertId={drawer.id} />
}

function StatusShell({
  fadeTab,
  selectTab,
  openInsights,
  animateDrawer
}: {
  fadeTab: OpsTab | null
  selectTab: (tab: OpsTab) => void
  openInsights: (target?: InsightsTarget) => void
  animateDrawer: boolean
}) {
  const { t } = useTranslation()
  const { view } = useOpsNav()
  const { paused, reduced, hidden } = useOpsMotion()
  const poll = useOpsPoll((signal) => opsApi.status(signal), [], STATUS_REFRESH_SECONDS)
  const overview = useOpsPoll((signal) => opsApi.overview(signal), [], OVERVIEW_REFRESH_SECONDS)
  const status = poll.data
  const shell = useMemo(() => ({ status, openInsights }), [status, openInsights])
  const Panel = PANELS[view.tab]

  // One page-level slot for rings (§2.12 rule 2); a non-baseline overview offers its candidates.
  const [slot] = useState(() => new AttentionSlot())
  const previousOverview = useRef<OpsOverview | null>(null)
  useEffect(() => {
    const prev = previousOverview.current
    const next = overview.data
    previousOverview.current = next
    if (!prev || !next || prev === next || overview.baseline || reduced || hidden || paused) return
    slot.offer(ringCandidates(prev, next), Date.now())
  }, [overview.data, overview.baseline, reduced, hidden, paused, slot])

  // "Activity (3)": what needs attention now, from the evaluator's snapshot (§2.2).
  const attention = overview.data?.attention
  const badges = useMemo(() => {
    const c = attentionCounts(attention)
    const open = c.critical + c.serious + c.warning
    return open ? { activity: open } : undefined
  }, [attention])

  const reloadStatus = poll.reload
  const reloadOverview = overview.reload
  const reloadAll = useCallback(() => {
    reloadStatus()
    reloadOverview()
  }, [reloadStatus, reloadOverview])

  return (
    <OpsShellContext.Provider value={shell}>
      <OpsOverviewContext.Provider value={overview}>
        <AttentionSlotContext.Provider value={slot}>
          <div className="ops-root relative h-full overflow-hidden" data-motion={reduced ? 'reduced' : undefined} data-paused={paused ? '' : undefined}>
            <div className="h-full overflow-y-auto">
              <div className="mx-auto flex max-w-[1360px] flex-col gap-4 px-4 pt-5 pb-14 md:px-7">
                <Header poll={poll} overview={overview} reloadAll={reloadAll} />
                {status && !status.available ? (
                  <Card className="flex max-w-[760px] flex-col gap-1">
                    <h2 className="m-0 text-[15px] font-semibold">{t('operations.unavailable.title')}</h2>
                    <p className="text-muted-foreground m-0 text-sm">{t('operations.unavailable.body')}</p>
                  </Card>
                ) : null}
                {poll.error && !status ? (
                  <Card>
                    <ErrorPanel error={poll.error} onRetry={poll.reload} />
                  </Card>
                ) : (
                  <>
                    {poll.error ? (
                      <Card className="py-2.5">
                        <ErrorPanel error={poll.error} onRetry={poll.reload} compact />
                      </Card>
                    ) : null}
                    <TabRow tab={view.tab} onSelect={selectTab} badges={badges} />
                    <OpsBanners />
                    <div
                      key={view.tab}
                      id={`ops-panel-${view.tab}`}
                      role="tabpanel"
                      aria-labelledby={`ops-tab-${view.tab}`}
                      className={fadeTab === view.tab ? 'ops-fade-in' : undefined}
                    >
                      <ViewBoundary resetKey={view.tab}>
                        <Panel />
                      </ViewBoundary>
                    </div>
                  </>
                )}
              </div>
            </div>
            <ViewBoundary resetKey={view.drawer ? writeOpsHash({ ...DEFAULT_OPS_HASH, drawer: view.drawer }) : 'none'}>
              <DrawerStack animateOpen={animateDrawer} render={renderDrawer} />
            </ViewBoundary>
            {actionsEnabled(status) ? <HostedActionDialogs /> : null}
          </div>
        </AttentionSlotContext.Provider>
      </OpsOverviewContext.Provider>
    </OpsShellContext.Provider>
  )
}

import { KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MotionConfig } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { PauseIcon, PlayIcon, RefreshCwIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { Card, ViewBoundary } from '@/features/TokenInsights/ui'
import ActivityTab from './ActivityTab'
import { OpsStatus, opsApi } from './api'
import ErrorPanel from './components/ErrorPanel'
import ProcessChip from './components/ProcessChip'
import RefreshHairline from './components/RefreshHairline'
import Skeleton from './components/Skeleton'
import { OpsShellContext } from './context'
import { ageSeconds, fmtAge, fmtClock } from './format'
import { DEFAULT_OPS_HASH, OPS_TABS, OpsHash, OpsTab, readOpsHash, writeOpsHash } from './hashState'
import HealthTab from './HealthTab'
import LogsTab from './LogsTab'
import { useOpsMotion } from './motion'
import OverviewTab from './OverviewTab'
import PerformanceTab from './PerformanceTab'
import { OpsPoll, useDocumentHidden, useOpsPoll } from './poll'
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

/** Re-renders a small part of the page every `ms`, so "updated 3 min ago" moves; idle while hidden. */
function useNow(ms: number) {
  const hidden = useDocumentHidden()
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (hidden) return
    const timer = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(timer)
  }, [ms, hidden])
  return now
}

function Freshness({ poll }: { poll: OpsPoll<OpsStatus> }) {
  const { t } = useTranslation()
  const now = useNow(15_000)
  if (!poll.updatedAt) return null
  const stale = isStale(poll.updatedAt, STATUS_REFRESH_SECONDS, now)
  return stale ? (
    <span className="text-[color:var(--ops-stale-fg)] rounded-md border border-dashed px-1.5 py-0.5 text-xs">
      {t('operations.stale', { age: fmtAge(ageSeconds(poll.updatedAt, now)) })}
    </span>
  ) : (
    <span className="text-muted-foreground font-mono text-xs">{t('operations.updated', { time: fmtClock(poll.updatedAt) })}</span>
  )
}

function Header({ poll }: { poll: OpsPoll<OpsStatus> }) {
  const { t } = useTranslation()
  const { paused, reduced, osReduced } = useOpsMotion()
  // The button shows the effective state. While the OS asks for reduced motion
  // the page cannot switch it back on, so the button is locked on.
  const prefs = useOpsPrefs()
  const status = poll.data
  return (
    <div className="relative flex flex-wrap items-center gap-x-4 gap-y-2.5">
      <h1 className="m-0 text-[26px] font-semibold tracking-tight">{t('operations.title')}</h1>
      <div className="flex flex-wrap items-center gap-2" aria-live="polite">
        {status ? <ProcessChip process={status.process} /> : poll.loading ? <Skeleton className="h-6 w-36" /> : null}
        {status?.version ? <span className="text-muted-foreground text-xs">{t('operations.version', { version: status.version })}</span> : null}
      </div>
      <div className="ms-auto flex flex-wrap items-center gap-2">
        <Freshness poll={poll} />
        <button
          type="button"
          onClick={poll.reload}
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
      <RefreshHairline slow={poll.slow} />
    </div>
  )
}

function TabRow({ tab, onSelect }: { tab: OpsTab; onSelect: (tab: OpsTab) => void }) {
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
        </button>
      ))}
    </div>
  )
}

function OperationsPage() {
  const [view, setView] = useState<OpsHash>(() => readOpsHash(window.location.hash) ?? DEFAULT_OPS_HASH)
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

  // The URL holds the view; leaving the tab drops it, so a reload lands where the user went.
  useEffect(() => {
    const next = writeOpsHash(view)
    if (window.location.hash !== next) window.history.replaceState(null, '', next)
  }, [view])
  useEffect(() => () => window.history.replaceState(null, '', '#/'), [])
  // A pasted link while the tab is open switches the view too.
  useEffect(() => {
    const onHash = () => {
      const next = readOpsHash(window.location.hash)
      if (next) setView(next)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // Radix portals render outside .ops-root; the page toggle reaches them through <html>.
  useEffect(() => {
    const root = document.documentElement
    if (reduceMotion) root.setAttribute('data-ops-motion', 'reduced')
    else root.removeAttribute('data-ops-motion')
    return () => root.removeAttribute('data-ops-motion')
  }, [reduceMotion])

  const selectTab = useCallback((tab: OpsTab) => {
    setView((v) => (v.tab === tab ? v : { ...v, tab, sub: null }))
    setFadeTab(tab)
  }, [])

  const openInsights = useCallback(() => useSettingsStore.getState().setCurrentTab('insights'), [])

  return (
    <OpsPrefsContext.Provider value={prefs}>
      {/* The first MotionConfig in the app, scoped to Operations: 'always' honours the page toggle. */}
      <MotionConfig reducedMotion={reduceMotion ? 'always' : 'user'}>
        <StatusShell view={view} fadeTab={fadeTab} selectTab={selectTab} openInsights={openInsights} />
      </MotionConfig>
    </OpsPrefsContext.Provider>
  )
}

function StatusShell({
  view,
  fadeTab,
  selectTab,
  openInsights
}: {
  view: OpsHash
  fadeTab: OpsTab | null
  selectTab: (tab: OpsTab) => void
  openInsights: () => void
}) {
  const { t } = useTranslation()
  const { paused, reduced } = useOpsMotion()
  const poll = useOpsPoll((signal) => opsApi.status(signal), [], STATUS_REFRESH_SECONDS)
  const status = poll.data
  const shell = useMemo(() => ({ status, openInsights }), [status, openInsights])
  const Panel = PANELS[view.tab]

  return (
    <OpsShellContext.Provider value={shell}>
      <div className="ops-root relative h-full overflow-hidden" data-motion={reduced ? 'reduced' : undefined} data-paused={paused ? '' : undefined}>
        <div className="h-full overflow-y-auto">
          <div className="mx-auto flex max-w-[1360px] flex-col gap-4 px-4 pt-5 pb-14 md:px-7">
            <Header poll={poll} />
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
                <TabRow tab={view.tab} onSelect={selectTab} />
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
      </div>
    </OpsShellContext.Provider>
  )
}

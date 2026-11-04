import { useCallback, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Seg } from '@/components/admin'
import { opsApi } from './api'
import { useOpsNav, useOpsShell } from './context'
import { OPS_SUBS } from './hashState'
import ConcurrencyView from './performance/ConcurrencyView'
import DepLatencyView from './performance/DepLatencyView'
import GpuView from './performance/GpuView'
import HttpView from './performance/HttpView'
import LlmView from './performance/LlmView'
import { DEFAULT_PERF_WINDOW, PERF_WINDOWS, PERF_WINDOW_SECONDS, PerfSection, PerfWindow, isPerfWindow, perfRefreshSeconds } from './performance/model'
import { PerfCtx } from './performance/chartKit'
import ThroughputView from './performance/ThroughputView'
import TopWorkspacesView from './performance/TopWorkspacesView'
import { useOpsPoll } from './poll'

/** The chosen window is a per-viewer convenience (localStorage, wrapped: blocked storage means the default). */
const WINDOW_KEY = 'LIGHTRAG-OPS-PERF-WINDOW'

const readWindow = (): PerfWindow => {
  try {
    const v = localStorage.getItem(WINDOW_KEY)
    return isPerfWindow(v) ? v : DEFAULT_PERF_WINDOW
  } catch {
    return DEFAULT_PERF_WINDOW
  }
}

const writeWindow = (w: PerfWindow) => {
  try {
    localStorage.setItem(WINDOW_KEY, w)
  } catch {
    // Storage blocked: the window holds for this page only.
  }
}

const SECTIONS = OPS_SUBS.performance as readonly PerfSection[]

/**
 * Performance (§2.7): a window [15m | 1h | 24h | 7d] and one section at a
 * time [LLM | Embed & Rerank | HTTP | Throughput | Concurrency | Top
 * workspaces | GPU]. GPU shows only while the server reads a GPU exporter
 * (OPS_GPU_EXPORTER_URL, D5). Only the visible section polls: every 60 s for 15 m and 1 h,
 * by hand for 24 h and 7 d. Event markers for the window come from /events
 * (deployments, provider rejections, admin actions) and are drawn on every
 * chart. Charts never animate. Cost and tokens stay in Token Insights.
 */
export default function PerformanceTab() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status } = useOpsShell()
  const sections = status?.features?.gpu === true ? SECTIONS : SECTIONS.filter((s) => s !== 'gpu')
  const section: PerfSection = nav.view.sub && sections.includes(nav.view.sub as PerfSection) ? (nav.view.sub as PerfSection) : 'llm'
  const [win, setWinState] = useState<PerfWindow>(readWindow)
  const [fade, setFade] = useState<PerfSection | null>(null)
  // A window change keeps the section mounted: the old data stays on screen until the new answer lands.
  const setWin = useCallback((w: PerfWindow) => {
    writeWindow(w)
    setWinState(w)
  }, [])
  const selectSection = (s: PerfSection) => {
    setFade(s)
    nav.setSub(s)
  }

  // Markers for the window; the windowless Concurrency section does not need them.
  const windowed = section !== 'concurrency'
  const events = useOpsPoll(
    (signal) => {
      const to = Date.now()
      return opsApi.events({ from: new Date(to - PERF_WINDOW_SECONDS[win] * 1000).toISOString(), to: new Date(to).toISOString() }, signal)
    },
    [win],
    perfRefreshSeconds(win),
    { enabled: windowed && status?.available !== false }
  )
  const ctx: PerfCtx = useMemo(() => ({ window: win, events: events.data ?? [] }), [win, events.data])

  const sectionLabel = (s: PerfSection) => t(`operations.perf.section.${s}`)
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-x-4 gap-y-2.5">
        <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">
          {t('operations.perf.window')}
          <Seg value={win} onChange={setWin} label={t('operations.perf.window')} options={PERF_WINDOWS.map((w) => ({ value: w, label: t(`operations.perf.windows.${w}`) }))} />
        </div>
        <div className="text-muted-foreground flex min-w-0 flex-col gap-0.5 text-xs">
          {t('operations.perf.sectionLabel')}
          <div className="overflow-x-auto">
            <Seg value={section} onChange={selectSection} label={t('operations.perf.sectionLabel')} options={sections.map((s) => ({ value: s, label: sectionLabel(s) }))} />
          </div>
        </div>
        <span className="text-muted-foreground pb-1.5 text-xs">
          {section === 'concurrency' ? t('operations.perf.nowNote') : perfRefreshSeconds(win) ? t('operations.perf.autoNote') : t('operations.perf.manualNote')}
        </span>
      </div>
      <div key={section} className={fade === section ? 'ops-fade-in' : undefined}>
        {section === 'llm' ? <LlmView ctx={ctx} /> : null}
        {section === 'deps' ? <DepLatencyView ctx={ctx} /> : null}
        {section === 'http' ? <HttpView ctx={ctx} /> : null}
        {section === 'throughput' ? <ThroughputView ctx={ctx} /> : null}
        {section === 'concurrency' ? <ConcurrencyView /> : null}
        {section === 'top' ? <TopWorkspacesView ctx={ctx} /> : null}
        {section === 'gpu' ? <GpuView ctx={ctx} /> : null}
      </div>
    </div>
  )
}

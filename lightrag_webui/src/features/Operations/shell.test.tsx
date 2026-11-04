import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { OpsStatus } from './api'

// The stores behind these components read localStorage as they load.
const memory = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, v),
    removeItem: (k: string) => void memory.delete(k)
  },
  configurable: true
})
const { OpsNavContext, OpsShellContext } = await import('./context')
const { DEFAULT_OPS_HASH } = await import('./hashState')
const { default: OverviewTab } = await import('./OverviewTab')
const { default: PerformanceTab } = await import('./PerformanceTab')
const { default: ProcessChip } = await import('./components/ProcessChip')
const { default: ErrorPanel } = await import('./components/ErrorPanel')
const { toOpsError } = await import('./errors')

const status = (patch: Partial<OpsStatus> = {}): OpsStatus => ({
  available: true,
  features: { durable: false, runs: false, logs: false, probes: false, alerts: false, insights: true, release_busy: false, gpu: false },
  process: { mode: 'uvicorn', expected_workers: 1, workers_live: null },
  operator: true,
  version: '1.5.1',
  generated_at: '2026-11-03T12:00:00Z',
  ...patch
})

const withShell = (node: React.ReactNode, s: OpsStatus | null) =>
  renderToString(<OpsShellContext.Provider value={{ status: s, openInsights: () => {} }}>{node}</OpsShellContext.Provider>)

describe('ProcessChip', () => {
  test('single process', () => {
    expect(renderToString(<ProcessChip process={{ mode: 'uvicorn', expected_workers: 1, workers_live: null }} />)).toContain('operations.process.uvicorn')
  })

  test('gunicorn before heartbeats exist shows the expected count without a tone', () => {
    const html = renderToString(<ProcessChip process={{ mode: 'gunicorn', expected_workers: 3, workers_live: null }} />)
    expect(html).toContain('operations.process.gunicorn')
    expect(html).not.toContain('ops-tone-')
  })

  test('gunicorn with a missing worker is a warning, with icon and text', () => {
    const html = renderToString(<ProcessChip process={{ mode: 'gunicorn', expected_workers: 3, workers_live: 2 }} />)
    expect(html).toContain('operations.process.gunicornLive')
    expect(html).toContain('ops-tone-warn')
    expect(html).toContain('operations.state.warning')
  })
})

describe('OverviewTab outside the shell', () => {
  test('without the overview poll it still shows what /status reports', () => {
    const html = withShell(<OverviewTab />, status())
    expect(html).toContain('operations.system.title')
    expect(html).toContain('operations.system.storageOk')
    expect(html).toContain('operations.system.workersPending')
    expect(html).toContain('operations.feature.multiWorkerOnly')
    expect(html).toContain('operations.system.singleProcessNote')
  })

  test('without PostgreSQL the storage row says so instead of failing', () => {
    expect(withShell(<OverviewTab />, status({ available: false }))).toContain('operations.system.storageMissing')
  })

  test('renders nothing before the first status answer', () => {
    expect(withShell(<OverviewTab />, null)).not.toContain('operations.system.title')
  })
})

describe('PerformanceTab (§2.7)', () => {
  const nav = (sub: string | null) => ({
    view: { ...DEFAULT_OPS_HASH, tab: 'performance' as const, sub },
    stack: [],
    move: 'none' as const,
    go: () => {},
    setSub: () => {},
    openDrawer: () => {},
    backDrawer: () => {},
    closeDrawer: () => {},
    setView: () => {}
  })
  const render = (sub: string | null, s: OpsStatus | null = status()) =>
    withShell(<OpsNavContext.Provider value={nav(sub)}>{<PerformanceTab />}</OpsNavContext.Provider>, s)

  test('the window and section controls; LLM is the default section, first paint is a skeleton', () => {
    const html = render(null)
    for (const w of ['15m', '1h', '24h', '7d']) expect(html).toContain(`operations.perf.windows.${w}`)
    for (const s of ['llm', 'deps', 'http', 'throughput', 'concurrency', 'top']) expect(html).toContain(`operations.perf.section.${s}`)
    expect(html).toContain('operations.perf.llm.title')
    expect(html).toContain('ops-skeleton')
    // The default window (1 h) refreshes by itself.
    expect(html).toContain('operations.perf.autoNote')
  })

  test('the section comes from the URL (osub); an unknown one falls back to LLM', () => {
    expect(render('concurrency')).toContain('operations.perf.concurrency.title')
    expect(render('concurrency')).toContain('operations.perf.nowNote')
    expect(render('http')).toContain('operations.perf.http.title')
    expect(render('bogus')).toContain('operations.perf.llm.title')
  })

  test('a remembered long window is manual', () => {
    memory.set('LIGHTRAG-OPS-PERF-WINDOW', '7d')
    try {
      const html = render('top')
      expect(html).toContain('operations.perf.manualNote')
      expect(html).toContain('operations.perf.top.title')
    } finally {
      memory.delete('LIGHTRAG-OPS-PERF-WINDOW')
    }
  })
})

describe('ErrorPanel', () => {
  const apiError = (code: number) => Object.assign(new Error(String(code)), { status: code, detail: 'x', requestId: '3c9e41d0' })

  test('a 403 asks to sign in again and offers no retry', () => {
    const html = renderToString(<ErrorPanel error={toOpsError(apiError(403))} onRetry={() => {}} />)
    expect(html).toContain('operations.error.forbidden')
    expect(html).toContain('operations.error.signInAgain')
    expect(html).not.toContain('operations.error.retry')
    expect(html).toContain('3c9e41d0')
  })

  test('a 500 shows the request id with a copy button and Retry', () => {
    const html = renderToString(<ErrorPanel error={toOpsError(apiError(500))} onRetry={() => {}} />)
    expect(html).toContain('role="alert"')
    expect(html).toContain('3c9e41d0')
    expect(html).toContain('operations.error.copyRef')
    expect(html).toContain('operations.error.retry')
  })
})

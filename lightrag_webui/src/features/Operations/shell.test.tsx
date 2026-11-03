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
const { OpsShellContext } = await import('./context')
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

describe('PerformanceTab', () => {
  test('links to Token Insights only when it records', () => {
    expect(withShell(<PerformanceTab />, status())).toContain('operations.openInsights')
    const off = status({ features: { ...status().features, insights: false } })
    expect(withShell(<PerformanceTab />, off)).not.toContain('operations.openInsights')
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

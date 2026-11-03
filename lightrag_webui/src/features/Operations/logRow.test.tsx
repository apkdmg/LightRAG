import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { LogEntry, OpsStatus } from './api'

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
const { LogRows } = await import('./components/LogRow')
const { RedactedText } = await import('./components/RedactedSpan')
const { OpsPrefsContext } = await import('./prefs')
const { OpsNavContext, OpsShellContext } = await import('./context')
const { DEFAULT_OPS_HASH } = await import('./hashState')
const { default: LogsTab } = await import('./LogsTab')
const { default: ExportDialog } = await import('./components/ExportDialog')
const { DEFAULT_LOG_FILTER } = await import('./logs')

const entry = (id: number, patch: Partial<LogEntry> = {}): LogEntry => ({
  id,
  ts: '2026-11-03T02:45:02.114Z',
  level: 40,
  logger: 'lightrag',
  module: 'pipeline',
  lineno: 2596,
  pid: 811,
  worker_id: null,
  workspace: 'fsktm',
  request_id: '3c9e41d0aa',
  client_request_id: null,
  run_id: '7f3a9c2e-1b2c-4d5e-8f90-123456789abc',
  track_id: null,
  doc_id: 'doc-9a2f00',
  operation: null,
  source: null,
  msg: 'Failed to extract for <email>: ProviderRejected 402\nsecond line',
  exc_type: 'ProviderRejected',
  has_exc: true,
  exc: null,
  fingerprint: 'b3a91c0e',
  repeat_count: 37,
  ...patch
})

const prefs = (patch: { reduceMotion?: boolean; paused?: boolean } = {}) => ({
  paused: false,
  setPaused: () => {},
  reduceMotion: false,
  setReduceMotion: () => {},
  ...patch
})

const nav = {
  view: { ...DEFAULT_OPS_HASH, tab: 'logs' as const, sub: 'stream' },
  stack: [],
  move: 'none' as const,
  go: () => {},
  setSub: () => {},
  openDrawer: () => {},
  backDrawer: () => {},
  closeDrawer: () => {},
  setView: () => {}
}

const render = (node: React.ReactNode, patch: { reduceMotion?: boolean; paused?: boolean } = {}) =>
  renderToString(
    <OpsPrefsContext.Provider value={prefs(patch)}>
      <OpsNavContext.Provider value={nav}>{node}</OpsNavContext.Provider>
    </OpsPrefsContext.Provider>
  )

const LOOP_CLASSES = ['ops-breathe', 'ops-flow', 'ops-ring-a', 'ops-ring-b', 'ops-pulse']
const rows = [entry(3), entry(2, { level: 30, msg: 'RateLimitError, retry 2/3 in 8s', has_exc: false, repeat_count: 1 }), entry(1, { logger: 'uvicorn.error', module: 'httptools_impl' })]

describe('LogRow batch under reduced motion (§2.12 "Follow-mode log rows")', () => {
  test('fresh rows appear at once: the marker stays, no fade, no transform, no loop', () => {
    const html = render(<LogRows items={rows} fresh={new Set([3, 2])} />, { reduceMotion: true })
    expect(html.match(/data-new=""/g)?.length).toBe(2)
    expect(html).not.toContain('ops-log-enter')
    expect(html).not.toContain('transform')
    expect(html).not.toContain('will-change')
    expect(LOOP_CLASSES.some((c) => html.includes(c))).toBe(false)
  })

  test('with motion allowed only the latest batch fades in, still without transforms', () => {
    const html = render(<LogRows items={rows} fresh={new Set([3])} />)
    expect(html.match(/ops-log-enter/g)?.length).toBe(1)
    expect(html.match(/data-new=""/g)?.length).toBe(1)
    expect(html).not.toContain('transform')
  })

  test('a repeated render of unchanged rows adds no animation classes (rule 1)', () => {
    const a = render(<LogRows items={rows} />)
    const b = render(<LogRows items={rows} />)
    expect(a).toBe(b)
    expect(a).not.toContain('ops-log-enter')
    expect(a).not.toContain('data-new')
  })

  test('"⏸ auto" adds no loop to rows', () => {
    const html = render(<LogRows items={rows} fresh={new Set([3])} />, { paused: true })
    expect(LOOP_CLASSES.some((c) => html.includes(c))).toBe(false)
  })
})

describe('LogRow content', () => {
  const html = render(<LogRows items={rows} />)

  test('level is icon plus text, and the line is a real link to the entry drawer', () => {
    expect(html).toContain('ERROR')
    expect(html).toContain('ops-tone-serious')
    expect(html).toContain('href="#/?ops=1&amp;otab=logs&amp;osub=stream&amp;olog=3"')
  })

  test('only the first line of the message shows, with masked spans marked', () => {
    expect(html).toContain('Failed to extract for ')
    expect(html).toContain('⟨email⟩')
    expect(html).not.toContain('second line')
    expect(html).toContain('▸ <!-- -->ProviderRejected')
    expect(html).toContain('×<!-- -->37')
  })

  test('lightrag rows show the module, other loggers their name', () => {
    expect(html).toContain('>pipeline<')
    expect(html).toContain('>uvicorn.error<')
  })

  test('context ids are pivots: run and doc open drawers, request filters the Logs tab', () => {
    expect(html).toContain('orun=7f3a9c2e-1b2c-4d5e-8f90-123456789abc')
    expect(html).toContain('odoc=fsktm%2Fdoc-9a2f00')
    expect(html).toContain('olq=w%3D7d%26req%3D3c9e41d0aa')
  })

  test('in a drawer rows are compact cards', () => {
    expect(render(<LogRows items={rows} inDrawer />)).toContain('ops-log-compact')
  })

  test('redacted text is escaped, never parsed as markup', () => {
    const out = renderToString(<RedactedText text={'<img src=x onerror=alert(1)> <ic>'} />)
    expect(out).not.toContain('<img')
    expect(out).toContain('&lt;img')
    expect(out).toContain('⟨ic⟩')
  })
})

describe('LogsTab', () => {
  const status = (patch: Partial<OpsStatus> = {}): OpsStatus => ({
    available: true,
    features: { logs: true },
    process: { mode: 'uvicorn', expected_workers: 1, workers_live: null },
    operator: true,
    version: '1.5.1',
    generated_at: '2026-11-03T12:00:00Z',
    ...patch
  })
  const withShell = (s: OpsStatus) =>
    render(
      <OpsShellContext.Provider value={{ status: s, openInsights: () => {} }}>
        <LogsTab />
      </OpsShellContext.Provider>
    )

  test('renders the filter bar, view switch, follow and export, and a skeleton before data', () => {
    const html = withShell(status())
    expect(html).toContain('operations.logs.level')
    expect(html).toContain('operations.logs.stream')
    expect(html).toContain('aria-pressed="true"')
    expect(html).toContain('operations.logs.followOn')
    expect(html).toContain('operations.logs.export')
    expect(html).toContain('ops-skeleton')
  })

  test('says when the log sink is off, read from features.log_sink', () => {
    // The shipped default: the store is up (logs: true) but OPS_LOG_SINK is off.
    expect(withShell(status({ features: { logs: true, log_sink: false } }))).toContain('operations.logs.sinkOff')
    expect(withShell(status({ features: { logs: true, log_sink: true } }))).not.toContain('operations.logs.sinkOff')
    // An unreachable store is not a sink that is off.
    expect(withShell(status({ features: { logs: false, log_sink: true } }))).not.toContain('operations.logs.sinkOff')
    expect(withShell(status())).not.toContain('operations.logs.sinkOff')
  })

  test('export is disabled for a non-operator', () => {
    const html = withShell(status({ operator: false }))
    expect(html).toMatch(/disabled=""[^>]*title="operations.export.operatorOnly"/)
  })

  test('the export dialog renders nothing while closed', () => {
    expect(renderToString(<ExportDialog open={false} onOpenChange={() => {}} filter={DEFAULT_LOG_FILTER} operator />)).toBe('')
  })
})

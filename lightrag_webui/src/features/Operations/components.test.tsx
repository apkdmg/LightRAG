import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { ActiveRun, AttentionItem, OpsOverview, OpsStatus } from './api'

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
const { default: KpiTile } = await import('./components/KpiTile')
const { default: AttentionList } = await import('./components/AttentionList')
const { default: ActiveRunsTable } = await import('./components/ActiveRunsTable')
const { RunningDot } = await import('./components/RunState')
const { OverallStateChip } = await import('./components/StatusChip')
const { default: StageTimeline } = await import('./components/StageTimeline')
const { default: StuckTable } = await import('./components/StuckTable')
const { default: Skeleton } = await import('./components/Skeleton')
const { OpsPrefsContext } = await import('./prefs')
const { OpsOverviewContext, OpsShellContext } = await import('./context')
const { default: OverviewTab, KpiRow } = await import('./OverviewTab')
const { DEFAULT_OPS_HASH } = await import('./hashState')
const { stuckRows } = await import('./processing/stuck')

const prefs = (patch: { reduceMotion?: boolean; paused?: boolean } = {}) => ({
  paused: false,
  setPaused: () => {},
  reduceMotion: false,
  setReduceMotion: () => {},
  ...patch
})

const withPrefs = (node: React.ReactNode, patch: { reduceMotion?: boolean; paused?: boolean } = {}) =>
  renderToString(<OpsPrefsContext.Provider value={prefs(patch)}>{node}</OpsPrefsContext.Provider>)

const LOOP_CLASSES = ['ops-breathe', 'ops-flow', 'ops-ring-a', 'ops-ring-b', 'ops-pulse']
const hasLoop = (html: string) => LOOP_CLASSES.some((c) => html.includes(c))

const to = { ...DEFAULT_OPS_HASH, tab: 'processing' as const, sub: 'backlog' }

describe('KpiTile', () => {
  test('the final value is on first paint, with an sr-only copy and the tween digits aria-hidden', () => {
    const html = withPrefs(<KpiTile label="Backlog" value={1412} state="ok" to={to} />)
    expect(html).toContain('<span aria-hidden="true">1,412</span>')
    expect(html).toContain('<span class="sr-only">1,412</span>')
  })

  test('is a real link to its drill-down', () => {
    const html = withPrefs(<KpiTile label="Backlog" value={3} state="ok" to={to} />)
    expect(html).toContain('href="#/?ops=1&amp;otab=processing&amp;osub=backlog"')
  })

  test('an OK tile names its state for screen readers only and shows its scope', () => {
    const html = withPrefs(<KpiTile label="Backlog" value={3} state="ok" to={to} />)
    expect(html).toContain('<span class="sr-only">operations.state.ok</span>')
    expect(html).toContain('operations.scope.host')
  })

  test('a worsened tile shows icon and text, not colour alone', () => {
    const html = withPrefs(<KpiTile label="Stuck" value={4} state="warning" to={to} />)
    expect(html).toContain('operations.state.warning')
    expect(html).toContain('<svg')
    expect(html).toContain('border-[color:var(--ops-warn)]')
  })

  test('a stale tile is never green and says it is stale', () => {
    const html = withPrefs(<KpiTile label="Backlog" value={3} state="ok" stale to={to} />)
    expect(html).toContain('operations.overall.stale')
    expect(html).toContain('ops-stale')
    expect(html).not.toContain('operations.state.ok')
  })

  test('a missing value reads "–", and a text value renders as given', () => {
    expect(withPrefs(<KpiTile label="LLM" value={null} state="unknown" to={to} />)).toContain('>–</span>')
    expect(withPrefs(<KpiTile label="Deps" display="11/12" state="warning" to={to} />)).toContain('11/12')
  })

  test('a worker-scoped value says which worker', () => {
    expect(withPrefs(<KpiTile label="x" value={1} state="ok" scope="worker 812" to={to} />)).toContain('operations.scope.worker')
  })

  test('first paint has no delta line, no tint and no inline transform', () => {
    const html = withPrefs(<KpiTile label="Backlog" value={530} state="warning" since={Date.now()} to={to} />)
    expect(html).not.toContain('operations.kpi.delta')
    expect(html).not.toContain('ops-tint')
    expect(html).not.toContain('transform')
  })
})

describe('AttentionList', () => {
  const items: AttentionItem[] = [
    { id: 'stale', rule: 'stale_docs', severity: 'warning', count: 27, target: { tab: 'processing', sub: 'stuck' } },
    { id: 'hung', rule: 'run_hung', severity: 'critical', workspace: 'fsktm', title: 'Run hung · fsktm', runbook: 'Cancel the run' }
  ]

  test('renders every item as a link to its drill-down', () => {
    const html = withPrefs(<AttentionList items={items} base={DEFAULT_OPS_HASH} empty="nothing" />)
    expect(html).toContain('href="#/?ops=1&amp;otab=processing&amp;osub=stuck"')
    expect(html).toContain('href="#/?ops=1&amp;otab=processing&amp;osub=stuck&amp;ows=fsktm"')
    expect((html.match(/<a /g) ?? []).length).toBe(2)
  })

  test('worst first, with severity icon and text and the runbook step', () => {
    const html = withPrefs(<AttentionList items={items} base={DEFAULT_OPS_HASH} empty="nothing" />)
    // Untranslated in tests, a rule title falls back to the rule id.
    expect(html.indexOf('Run hung')).toBeGreaterThan(-1)
    expect(html.indexOf('Run hung')).toBeLessThan(html.indexOf('stale_docs'))
    expect(html).toContain('operations.state.critical')
    expect(html).toContain('operations.attention.next')
  })

  test('says what was checked when nothing needs attention', () => {
    const html = withPrefs(<AttentionList items={[]} base={DEFAULT_OPS_HASH} empty="No issues. Checked 3,012 workspaces" />)
    expect(html).toContain('No issues. Checked 3,012 workspaces')
    expect(html).not.toContain('<a ')
  })

  test('first paint does not slide items in, and no ring plays', () => {
    const html = withPrefs(<AttentionList items={items} base={DEFAULT_OPS_HASH} empty="nothing" />)
    expect(html).not.toContain('translateY')
    expect(hasLoop(html)).toBe(false)
  })
})

describe('motion under reduced motion, pause and staleness (§2.12 rules 1, 3)', () => {
  const runs: ActiveRun[] = [
    { workspace: 'space1', kind: 'email_batch', state: 'running', started_at: '2026-11-03T04:01:00Z', docs_done: 6, docs_total: 10 },
    { workspace: 'hr-office', kind: 'process', state: 'paused', pause: { status_code: 402, slot: 'extract' }, docs_done: 1, docs_total: 3 }
  ]
  const now = Date.parse('2026-11-03T04:04:00Z')

  test('the running dot breathes only when motion is allowed', () => {
    expect(withPrefs(<RunningDot />)).toContain('ops-breathe')
    expect(withPrefs(<RunningDot />, { reduceMotion: true })).not.toContain('ops-breathe')
    expect(withPrefs(<RunningDot />, { paused: true })).not.toContain('ops-breathe')
    expect(withPrefs(<RunningDot stale />)).not.toContain('ops-breathe')
  })

  test('active runs under reduced motion: final values, progress text, no loops', () => {
    const html = withPrefs(<ActiveRunsTable runs={runs} now={now} />, { reduceMotion: true })
    expect(hasLoop(html)).toBe(false)
    expect(html).toContain('role="progressbar"')
    expect(html).toContain('6<!-- -->/<!-- -->10')
    expect(html).toContain('3m')
    expect(html).toContain('operations.active.pausedCode')
  })

  test('"⏸ auto" with a stale panel stops the loops', () => {
    expect(hasLoop(withPrefs(<ActiveRunsTable runs={runs} now={now} stale />, { paused: true }))).toBe(false)
  })

  test('a repeated render with unchanged props adds no animation classes', () => {
    const a = withPrefs(<ActiveRunsTable runs={runs} now={now} />)
    const b = withPrefs(<ActiveRunsTable runs={runs} now={now} />)
    expect(a).toBe(b)
    expect(a).not.toContain('ops-ring')
  })

  test('the skeleton is a sized block; its pulse is capped in CSS, and reduced motion stops it there', () => {
    expect(renderToString(<Skeleton className="h-6 w-40" />)).toContain('ops-skeleton')
  })

  test('the state chip carries icon and text, and no ring on first paint', () => {
    const html = withPrefs(<OverallStateChip state="critical" word="degraded" counts={{ critical: 1, serious: 0, warning: 2 }} />)
    expect(html).toContain('operations.overall.degraded')
    expect(html).toContain('operations.overall.count.critical')
    expect(html).toContain('operations.overall.count.warning')
    expect(html).not.toContain('operations.overall.count.serious')
    expect(hasLoop(html)).toBe(false)
  })

  test('a stale state chip is not green', () => {
    const html = withPrefs(<OverallStateChip state="ok" word="healthy" counts={{ critical: 0, serious: 0, warning: 0 }} stale />)
    expect(html).toContain('ops-tone-unknown')
    expect(html).not.toContain('ops-tone-ok')
  })
})

describe('StageTimeline and StuckTable', () => {
  test('stages in order, a failed stage with icon and text', () => {
    const html = renderToString(
      <StageTimeline
        stages={[
          { stage: 'enqueued', started_at: '2026-11-03T01:58:00Z' },
          { stage: 'parsing', duration_ms: 400 },
          { stage: 'processing', state: 'failed' }
        ]}
      />
    )
    expect(html.indexOf('enqueued')).toBeGreaterThan(-1)
    expect(html.indexOf('enqueued')).toBeLessThan(html.indexOf('parsing'))
    expect(html).toContain('400 ms')
    expect(html).toContain('operations.stage.failedState')
  })

  test('stale docs link to their document drawer with the masked name', () => {
    const rows = stuckRows(
      {
        orphaned: [],
        hung: [],
        busy_without_run: [],
        reservations: [],
        stale_docs: { items: [{ workspace: 'library', doc_id: 'doc-77c0', status: 'processing', file_path_masked: 'R*****.pdf', age_s: 4320 }], total: 1 },
        thresholds: {}
      },
      false
    )
    const html = renderToString(<StuckTable rows={rows} />)
    expect(html).toContain('R*****.pdf')
    expect(html).toContain('odoc=library%2Fdoc-77c0')
    expect(html).toContain('1h 12m')
  })
})

describe('OverviewTab', () => {
  const status: OpsStatus = {
    available: true,
    features: { durable: true, runs: false, logs: false, probes: false, alerts: false, insights: true, release_busy: false, gpu: false },
    process: { mode: 'uvicorn', expected_workers: 1, workers_live: null },
    operator: true,
    version: '1.5.1',
    generated_at: '2026-11-03T04:00:00Z'
  }
  const data: OpsOverview = {
    state: 'warning',
    worst: 'warning',
    kpis: {
      active: { count: 3, longest_s: 2520, approximate: true },
      backlog: { inflight: 412 },
      failed_24h: { real: 17, cancelled: 5, dup: 812 },
      stuck: { runs: 0, docs: 27, reservations: 0 },
      email_queue: { records: 38, longest_pending_s: 2460 }
    },
    attention: [{ id: 'stale', rule: 'stale_docs', severity: 'warning', count: 27 }],
    checked: { workspaces: 3012, active: 3, email_queue: true, at: '2026-11-03T04:04:31Z' },
    active_top: [],
    deps: []
  }
  const poll = { data, error: null, loading: false, slow: false, updatedAt: Date.now(), previousAt: null, baseline: true, reload: () => {} }

  const render = (node: React.ReactNode) =>
    withPrefs(
      <OpsShellContext.Provider value={{ status, openInsights: () => {} }}>
        <OpsOverviewContext.Provider value={poll}>{node}</OpsOverviewContext.Provider>
      </OpsShellContext.Provider>
    )

  test('eight KPI tiles, each a link, with the dup and cancelled counts in text', () => {
    const html = render(<KpiRow data={data} poll={poll} />)
    expect((html.match(/class="ops-kpi /g) ?? []).length).toBe(8)
    expect(html).toContain('operations.kpi.failed.cancelled')
    expect(html).toContain('operations.kpi.failed.dup')
    expect(html).toContain('osub=failures')
  })

  test('unknown tiles (deps, LLM before their data exists) are not green', () => {
    const html = render(<KpiRow data={data} poll={poll} />)
    expect(html).toContain('operations.kpi.deps.later')
    expect(html).toContain('operations.state.unknown')
  })

  test('single-process mode reads "1 process" on the server tile', () => {
    expect(render(<KpiRow data={data} poll={poll} />)).toContain('operations.kpi.workers.single')
  })

  test('renders the attention list, the server card and the processing panel', () => {
    const html = render(<OverviewTab />)
    expect(html).toContain('operations.attention.title')
    expect(html).toContain('stale_docs')
    expect(html).toContain('osub=stuck')
    expect(html).toContain('operations.system.title')
    expect(html).toContain('operations.overview.processingNow')
    expect(html).toContain('operations.approximate.label')
  })

  test('a server without PostgreSQL gets a note, not KPI tiles', () => {
    const html = withPrefs(
      <OpsShellContext.Provider value={{ status: { ...status, available: false }, openInsights: () => {} }}>
        <OpsOverviewContext.Provider value={{ ...poll, data: { ...data, available: false } }}>
          <OverviewTab />
        </OpsOverviewContext.Provider>
      </OpsShellContext.Provider>
    )
    expect(html).not.toContain('class="ops-kpi ')
    expect(html).toContain('operations.unavailableShort')
  })
})
